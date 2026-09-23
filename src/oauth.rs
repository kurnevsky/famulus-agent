//! Signing in to an MCP server that asks for it.
//!
//! A server behind a login answers the first request with a 401, and what
//! follows is OAuth as the protocol lays it out: find where the server signs
//! people in, register with it (or use the client it was given in the file),
//! send the user's browser there, and take back the code it returns on a port
//! of this machine. rmcp does the protocol; what is here is the waiting for a
//! person, and the keeping of what they signed in to.
//!
//! What a sign-in brings back is kept in the system keyring, under the
//! server's URL, so the next session starts signed in and rmcp refreshes the
//! token when it runs out. A keyring that cannot be reached is not a server
//! that cannot be reached: the login lasts the session, and a note says so.
//!
//! All of it happens as the servers come up, before the terminal is taken
//! over, so the address to open and what is being waited for are plain lines
//! on stderr.
//!
//! ```toml
//! [linear]
//! url = "https://mcp.linear.app/mcp"
//!
//! # A server that does not register clients itself is given one.
//! [github]
//! url = "https://api.githubcopilot.com/mcp/"
//! oauth.client-id = "Iv1.0123456789abcdef"
//! oauth.client-secret-command = "pass show github/fa-oauth"
//! oauth.port = 8765
//! ```

use std::net::Ipv4Addr;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use rmcp::service::ClientInitializeError;
use rmcp::transport::auth::{
  AuthError, AuthorizationCallback, AuthorizationManager, CredentialStore, InMemoryCredentialStore, OAuthClientConfig,
  StoredCredentials,
};
use rmcp::transport::streamable_http_client::StreamableHttpError;
use serde::Deserialize;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, TcpStream};

use crate::keyring;

/// How long a person has to sign in. Longer than anything the network is
/// given, since it is someone finding a password rather than a server
/// finding an answer; short enough that a browser that never opened is not a
/// session that never starts.
const LOGIN_TIMEOUT: Duration = Duration::from_secs(300);

/// How long the browser has to say what it came back with, once it has
/// connected. A browser opens connections it never uses, and one of those is
/// not to hold up the one that carries the code.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(10);

/// Where on this machine the browser is sent back to.
const CALLBACK: &str = "/callback";

/// What the keyring knows these entries by.
const APPLICATION: &str = "famulus-agent";

/// The name a server that registers clients itself is told this one goes by.
const CLIENT_NAME: &str = "fa";

/// How a server's login is set up, for a server that does not register
/// clients itself or needs to be asked for something in particular. A server
/// that does need nothing here: signing in is what a 401 starts.
#[derive(Debug, Default, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields, rename_all = "kebab-case")]
pub struct Settings {
  /// The client registered with the server by hand, for one that registers
  /// no one itself.
  pub client_id: Option<String>,
  /// Its secret, if it has one.
  pub client_secret: Option<String>,
  /// The same, for a secret that should not sit in a file: a line of shell
  /// whose output is the secret.
  pub client_secret_command: Option<String>,
  /// What to ask for, where what the server says it offers is not it.
  pub scopes: Vec<String>,
  /// The port the browser comes back to. Any free one, unless the client
  /// was registered with an address that names one.
  pub port: Option<u16>,
}

/// Whether the server turned the connection down for want of a login, or of
/// a login that still holds.
pub fn wants_login(err: &ClientInitializeError) -> bool {
  let ClientInitializeError::TransportError { error, .. } = err else {
    return false;
  };
  matches!(
    error.error.downcast_ref::<StreamableHttpError<reqwest::Error>>(),
    Some(
      StreamableHttpError::AuthRequired(_)
        | StreamableHttpError::Auth(
          AuthError::AuthorizationRequired | AuthError::TokenExpired | AuthError::TokenRefreshFailed(_)
        )
    )
  )
}

/// The client a signed-in server is called through: rmcp's own, with the
/// token put on every request. Redirects are not followed, so the token is
/// never sent anywhere the server did not answer from.
pub fn client(manager: AuthorizationManager) -> rmcp::transport::auth::AuthClient<reqwest::Client> {
  let http = reqwest::Client::builder()
    .pool_max_idle_per_host(0)
    .redirect(reqwest::redirect::Policy::none())
    .build()
    .expect("a client with nothing to fail on");
  rmcp::transport::auth::AuthClient::new(http, manager)
}

/// The login the keyring has for `url`, set up to be used, or `None` when
/// there is none to use.
pub async fn resume(url: &str, settings: &Settings, store: &Store) -> Result<Option<AuthorizationManager>> {
  let mut manager = AuthorizationManager::new(url).await?;
  manager.set_credential_store(store.clone());
  if !manager.initialize_from_store().await? {
    return Ok(None);
  }
  // What the keyring keeps is the client's name and not its secret, and a
  // token is refreshed as the client it was given to.
  if let Some(secret) = secret(settings).await?
    && let Some(id) = &settings.client_id
  {
    manager.configure_client(OAuthClientConfig::new(id, url).with_client_secret(secret))?;
  }
  Ok(Some(manager))
}

/// Sign in to the server `name` at `url`, with the user's browser.
pub async fn login(name: &str, url: &str, settings: &Settings, store: &Store) -> Result<AuthorizationManager> {
  let secret = secret(settings).await?;
  let mut manager = AuthorizationManager::new(url).await?;
  manager.set_credential_store(store.clone());
  let metadata = manager
    .discover_metadata()
    .await
    .context("could not find where it signs people in")?;
  manager.set_metadata(metadata);

  let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, settings.port.unwrap_or(0)))
    .await
    .context("could not listen for the browser to come back")?;
  let redirect = format!("http://127.0.0.1:{}{CALLBACK}", listener.local_addr()?.port());

  // What the file asks for, else what the server says it offers — with a
  // refresh token asked for too, where it offers those.
  let asked = settings.scopes.join(" ");
  let scopes = manager.select_scopes((!asked.is_empty()).then_some(asked.as_str()), &[]);
  let scopes: Vec<&str> = scopes.iter().map(String::as_str).collect();
  match &settings.client_id {
    Some(id) => {
      let mut client = OAuthClientConfig::new(id, &redirect);
      if let Some(secret) = secret {
        client = client.with_client_secret(secret);
      }
      manager.configure_client(client)?;
    }
    None => {
      manager
        .register_client(CLIENT_NAME, &redirect, &scopes)
        .await
        .context("could not register with it (a client-id in its oauth table is the way round that)")?;
    }
  }
  let address = manager.get_authorization_url(&scopes).await?;

  eprintln!("MCP {name} wants you to sign in. Open this, if a browser does not:\n\n  {address}\n");
  eprintln!("Waiting for the sign-in to finish…");
  let opening = open(&address);
  let back = tokio::time::timeout(LOGIN_TIMEOUT, callback(listener, name)).await;
  // An opener still going is left to go on; tokio reaps it when it ends.
  opening.abort();
  let back = back.map_err(|_| anyhow::anyhow!("no sign-in in {}s", LOGIN_TIMEOUT.as_secs()))??;
  let answer = AuthorizationCallback::from_redirect_url(&back)?;
  manager
    .exchange_code_for_token_with_issuer(&answer.code, &answer.csrf_token, answer.issuer.as_deref())
    .await
    .context("could not trade the sign-in for a token")?;
  eprintln!("Signed in to {name}.");
  Ok(manager)
}

/// The client secret, however the file gives it.
async fn secret(settings: &Settings) -> Result<Option<String>> {
  crate::mcp::one(
    settings.client_secret.as_ref(),
    settings.client_secret_command.as_ref(),
    "client-secret",
  )
  .await
}

/// Hand the address to whatever opens addresses here. Only a convenience:
/// the address is on the screen, and a machine with no browser is one where
/// it is opened somewhere else — so an opener that could not is said, for the
/// address to be copied, and is not a reason to stop waiting.
///
/// What is handed back watches the opener, and is to be stopped once the
/// wait is over: some openers only finish when the browser does, and by then
/// the terminal is the session's and not for printing on.
fn open(address: &str) -> tokio::task::JoinHandle<()> {
  let opener = match std::env::consts::OS {
    "macos" => "open",
    _ => "xdg-open",
  };
  let address = address.to_string();
  tokio::spawn(async move {
    if let Some(failed) = opened(opener, &address).await {
      eprintln!("No browser opened ({failed}): open the address above yourself.");
    }
  })
}

/// Run `opener` on `address`, and what went wrong with it, if anything did.
async fn opened(opener: &str, address: &str) -> Option<String> {
  let spawned = tokio::process::Command::new(opener)
    .arg(address)
    .stdin(std::process::Stdio::null())
    .stdout(std::process::Stdio::null())
    .stderr(std::process::Stdio::piped())
    .spawn();
  let child = match spawned {
    Ok(child) => child,
    Err(err) => return Some(format!("could not run {opener}: {err}")),
  };
  match child.wait_with_output().await {
    Err(err) => Some(format!("{opener}: {err}")),
    Ok(output) if output.status.success() => None,
    // Its own first line, which is usually the reason: no display, no
    // browser, no idea what opens this.
    Ok(output) => Some(match String::from_utf8_lossy(&output.stderr).lines().next() {
      Some(said) if !said.trim().is_empty() => format!("{opener}: {}", said.trim()),
      _ => format!("{opener} {}", output.status),
    }),
  }
}

// ---------------------------------------------------------------- the way back

/// Wait for the browser to come back to `listener`, and give it a page that
/// says how it went. What comes back is the whole address it was sent to,
/// for rmcp to read the code out of; a server that said no is an error with
/// its reason.
async fn callback(listener: TcpListener, name: &str) -> Result<String> {
  let origin = format!("http://{}", listener.local_addr()?);
  // Every connection on its own, since the one with the code need not be the
  // first: a browser opens a spare, and asks for an icon. Those still being
  // read once it has come go with the set.
  let mut reading = tokio::task::JoinSet::new();
  loop {
    let (target, stream) = tokio::select! {
      accepted = listener.accept() => {
        let (stream, _) = accepted.context("stopped listening for the browser")?;
        reading.spawn(tokio::time::timeout(REQUEST_TIMEOUT, target(stream)));
        continue;
      }
      Some(read) = reading.join_next() => match read {
        Ok(Ok(Some(read))) => read,
        _ => continue,
      },
    };
    let address = format!("{origin}{target}");
    let Ok(url) = url::Url::parse(&address) else {
      continue;
    };
    if url.path() != CALLBACK {
      reply(stream, "404 Not Found", "Nothing here.").await;
      continue;
    }
    let said = |key: &str| url.query_pairs().find(|(k, _)| k == key).map(|(_, v)| v.into_owned());
    if let Some(error) = said("error") {
      let reason = match said("error_description") {
        Some(description) => format!("{error}: {description}"),
        None => error,
      };
      reply(stream, "200 OK", &format!("{name} did not sign you in: {reason}")).await;
      bail!("the sign-in was turned down: {reason}");
    }
    reply(
      stream,
      "200 OK",
      &format!("Signed in to {name}. This tab can be closed."),
    )
    .await;
    return Ok(address);
  }
}

/// What one connection asks for — the path and query of its request line —
/// and the connection, to answer it on.
async fn target(stream: TcpStream) -> Option<(String, TcpStream)> {
  let mut reader = BufReader::new(stream);
  let mut line = String::new();
  // The request line is all that is wanted, and it comes first; a request
  // that has sent a lot without one is not a browser coming back.
  (&mut reader).take(16 * 1024).read_line(&mut line).await.ok()?;
  if !line.ends_with('\n') {
    return None;
  }
  let mut parts = line.split_whitespace();
  let (method, target) = (parts.next()?, parts.next()?);
  (method == "GET" && target.starts_with('/')).then(|| (target.to_string(), reader.into_inner()))
}

async fn reply(mut stream: TcpStream, status: &str, text: &str) {
  let body = format!(
    "<!doctype html><meta charset=utf-8><title>fa</title><p style=\"font:1.2em sans-serif;margin:3em\">{}</p>",
    escape(text)
  );
  let response = format!(
    "HTTP/1.1 {status}\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
    body.len()
  );
  let _ = stream.write_all(response.as_bytes()).await;
  let _ = stream.shutdown().await;
}

/// Text as HTML shows it, since part of it is what the server sent.
fn escape(text: &str) -> String {
  text
    .replace('&', "&amp;")
    .replace('<', "&lt;")
    .replace('>', "&gt;")
    .replace('"', "&quot;")
}

// ---------------------------------------------------------------- the keyring

/// Where one server's login is kept: the keyring, under its URL, with a copy
/// for the session in case the keyring is not there to keep it.
#[derive(Clone)]
pub struct Store {
  url: String,
  session: InMemoryCredentialStore,
  /// What went wrong with the keyring, to say once the server is up.
  trouble: Arc<Mutex<Option<String>>>,
}

impl Store {
  pub fn new(url: &str) -> Self {
    Self {
      url: url.to_string(),
      session: InMemoryCredentialStore::new(),
      trouble: Arc::default(),
    }
  }

  /// Why the login will not outlast the session, if it will not.
  pub fn trouble(&self) -> Option<String> {
    self.trouble.lock().ok()?.clone()
  }

  fn note(&self, err: impl std::fmt::Display) {
    if let Ok(mut trouble) = self.trouble.lock() {
      trouble.get_or_insert_with(|| format!("the login will not be remembered: {err}"));
    }
  }

  fn attributes(&self) -> [(&str, &str); 2] {
    [("application", APPLICATION), ("mcp-url", &self.url)]
  }

  async fn read(&self) -> Result<Option<StoredCredentials>> {
    let keyring = keyring::shared().await.map_err(anyhow::Error::msg)?;
    let items = keyring.search_items(&self.attributes()).await?;
    let Some(item) = items.first() else {
      return Ok(None);
    };
    let secret = keyring::secret(item).await?;
    // An entry this cannot read is one from some other version of it, and
    // the answer to that is signing in again, not refusing to start.
    Ok(serde_json::from_slice(secret.as_bytes()).ok())
  }

  async fn write(&self, credentials: &StoredCredentials) -> Result<()> {
    let keyring = keyring::shared().await.map_err(anyhow::Error::msg)?;
    let secret = serde_json::to_vec(credentials)?;
    let label = format!("fa: MCP login for {}", self.url);
    Ok(keyring.create_item(&label, &self.attributes(), secret, true).await?)
  }
}

#[async_trait::async_trait]
impl CredentialStore for Store {
  async fn load(&self) -> Result<Option<StoredCredentials>, AuthError> {
    // What this session was given beats what the keyring had, which it
    // replaced.
    if let Some(credentials) = self.session.load().await? {
      return Ok(Some(credentials));
    }
    match self.read().await {
      Ok(Some(credentials)) => {
        self.session.save(credentials.clone()).await?;
        Ok(Some(credentials))
      }
      Ok(None) => Ok(None),
      Err(err) => {
        self.note(err);
        Ok(None)
      }
    }
  }

  async fn save(&self, credentials: StoredCredentials) -> Result<(), AuthError> {
    if let Err(err) = self.write(&credentials).await {
      self.note(err);
    }
    self.session.save(credentials).await
  }

  async fn clear(&self) -> Result<(), AuthError> {
    if let Ok(keyring) = keyring::shared().await
      && let Err(err) = keyring.delete(&self.attributes()).await
    {
      self.note(err);
    }
    self.session.clear().await
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  async fn listening() -> (TcpListener, String) {
    let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).await.expect("a port");
    let origin = format!("http://{}", listener.local_addr().expect("an address"));
    (listener, origin)
  }

  /// What a browser does: ask for `target`, and read the page it is given.
  async fn visit(origin: &str, target: &str) -> String {
    let address = origin.trim_start_matches("http://");
    let mut stream = TcpStream::connect(address).await.expect("a connection");
    let request = format!("GET {target} HTTP/1.1\r\nHost: {address}\r\n\r\n");
    stream.write_all(request.as_bytes()).await.expect("sent");
    let mut page = String::new();
    stream.read_to_string(&mut page).await.expect("a page");
    page
  }

  #[tokio::test]
  async fn the_browser_coming_back_is_the_address_it_came_back_to() {
    let (listener, origin) = listening().await;
    let waiting = tokio::spawn(async move { callback(listener, "docs").await });
    // A connection that says nothing, and a request for something else, do
    // not stand in the way of the one that carries the code.
    let idle = TcpStream::connect(origin.trim_start_matches("http://"))
      .await
      .expect("a connection");
    let icon = visit(&origin, "/favicon.ico").await;
    assert!(icon.starts_with("HTTP/1.1 404"), "{icon}");
    let page = visit(&origin, "/callback?code=abc&state=xyz").await;
    assert!(page.starts_with("HTTP/1.1 200"), "{page}");
    assert!(page.contains("Signed in to docs"), "{page}");
    let back = waiting.await.expect("joined").expect("the address");
    assert_eq!(back, format!("{origin}/callback?code=abc&state=xyz"));
    let answer = AuthorizationCallback::from_redirect_url(&back).expect("a code");
    assert_eq!((answer.code.as_str(), answer.csrf_token.as_str()), ("abc", "xyz"));
    drop(idle);
  }

  #[tokio::test]
  async fn a_server_that_said_no_is_said_with_its_reason() {
    let (listener, origin) = listening().await;
    let waiting = tokio::spawn(async move { callback(listener, "docs").await });
    let page = visit(&origin, "/callback?error=access_denied&error_description=%3Cnope%3E").await;
    assert!(
      page.contains("did not sign you in: access_denied: &lt;nope&gt;"),
      "{page}"
    );
    let err = waiting.await.expect("joined").expect_err("turned down").to_string();
    assert!(err.contains("access_denied: <nope>"), "{err}");
  }

  #[tokio::test]
  async fn an_opener_that_could_not_open_says_why() {
    assert_eq!(opened("true", "https://example.com").await, None);
    let missing = opened("fa-no-such-opener", "https://example.com")
      .await
      .expect("not there");
    assert!(missing.starts_with("could not run fa-no-such-opener"), "{missing}");
    // What it said for itself, rather than only that it failed.
    let failed = opened("bash", "fa-no-such-script").await.expect("failed");
    assert!(
      failed.starts_with("bash: ") && failed.contains("fa-no-such-script"),
      "{failed}"
    );
    let silent = opened("false", "https://example.com").await.expect("failed");
    assert_eq!(silent, "false exit status: 1");
  }

  #[tokio::test]
  async fn a_secret_is_given_one_way_or_the_other() {
    let given = |secret: Option<&str>, command: Option<&str>| Settings {
      client_secret: secret.map(str::to_string),
      client_secret_command: command.map(str::to_string),
      ..Settings::default()
    };
    assert_eq!(secret(&given(None, None)).await.expect("none"), None);
    assert_eq!(
      secret(&given(Some("s"), None)).await.expect("written"),
      Some("s".into())
    );
    assert_eq!(
      secret(&given(None, Some("echo t"))).await.expect("printed"),
      Some("t".into())
    );
    let err = secret(&given(Some("s"), Some("echo t"))).await.expect_err("twice");
    assert!(err.to_string().contains("both"), "{err}");
  }
}
