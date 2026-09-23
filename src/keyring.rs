//! The system keyring, through `oo7`: the Secret Service on the session bus,
//! or the portal-backed file inside a sandbox.
//!
//! One connection serves the session, opened the first time something asks
//! for it. It holds what an MCP sign-in brought back, and a token the user
//! keeps there themselves for a server's `token-keyring` to name.

use std::collections::BTreeMap;

use anyhow::{Result, anyhow, bail};
use tokio::sync::OnceCell;

/// The keyring, or why there is none.
static KEYRING: OnceCell<Result<oo7::Keyring, String>> = OnceCell::const_new();

/// The keyring, opened once for everything that asks.
pub async fn shared() -> Result<&'static oo7::Keyring, String> {
  KEYRING
    .get_or_init(|| async { oo7::Keyring::new().await.map_err(|err| err.to_string()) })
    .await
    .as_ref()
    .map_err(Clone::clone)
}

/// What `item` holds, unlocked first if it has to be — which is the keyring's
/// own dialog, not a question in this terminal.
pub async fn secret(item: &oo7::Item) -> Result<oo7::Secret, String> {
  if item.is_locked().await.map_err(|e| e.to_string())? {
    item.unlock().await.map_err(|e| e.to_string())?;
  }
  item.secret().await.map_err(|e| e.to_string())
}

/// The text the keyring holds under exactly these attributes, without the
/// newline a secret pasted in tends to carry.
///
/// Exactly one item: none is a name that is wrong, and several are a choice
/// there is no rule for, so both are said rather than guessed past.
pub async fn lookup(attributes: &BTreeMap<String, String>) -> Result<String> {
  let named = attributes
    .iter()
    .map(|(key, value)| format!("{key}={value}"))
    .collect::<Vec<_>>()
    .join(" ");
  let keyring = shared().await.map_err(|err| anyhow!("no keyring: {err}"))?;
  let items = keyring
    .search_items(attributes)
    .await
    .map_err(|err| anyhow!("could not search the keyring: {err}"))?;
  let item = match items.as_slice() {
    [item] => item,
    [] => bail!("nothing in the keyring under {named}"),
    many => bail!(
      "{} items in the keyring under {named}, and no telling which",
      many.len()
    ),
  };
  let secret = secret(item).await.map_err(|err| anyhow!("{named}: {err}"))?;
  let text = std::str::from_utf8(secret.as_bytes()).map_err(|_| anyhow!("{named}: not text"))?;
  Ok(text.trim_end().to_string())
}
