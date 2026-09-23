//! Settings kept in a file rather than typed on every start.
//!
//! `config.toml` sits where `mcp.toml` does, on the XDG search path and
//! nowhere else, and says what the flags say: each key is a long flag's name,
//! without the dashes in front of it. A flag or its variable, when given, has
//! the last word, so the file is only ever the default.
//!
//! ```toml
//! provider = "openai"
//! base-url = "http://localhost:9931/v1"
//! model = "Qwen3.8-27B-Q4_K_M.gguf"
//! model = "qwen2.5-coder"
//! no-bell = true
//! ```
//!
//! The key is the one value with a command form, as `env-command` is for an MCP
//! server: a key is better kept out of the file it is used from.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Deserializer};

use crate::agent::Provider;
use crate::ui::ScrollbarMode;

/// The name settings are kept under, in each configuration directory.
pub const FILE: &str = "config.toml";

/// How long a command that produces a value has to produce it.
const VALUE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

/// What the files say, merged: a key the nearer file gives is the one used.
///
/// A key that is not one of these is an error naming the line it is on, the
/// way it is in `mcp.toml`: a misspelled `modle` that went quietly would be a
/// setting that never took for no stated reason.
#[derive(Debug, Default, Deserialize, PartialEq)]
#[serde(deny_unknown_fields, rename_all = "kebab-case")]
pub struct Settings {
  #[serde(default, deserialize_with = "value_enum")]
  pub provider: Option<Provider>,
  pub base_url: Option<String>,
  pub api_key: Option<String>,
  /// A line of shell whose output is the key, run once at start and only if
  /// nothing nearer gave one.
  pub api_key_command: Option<String>,
  pub model: Option<String>,
  pub system_prompt: Option<String>,
  pub max_tokens: Option<u64>,
  pub context_window: Option<u64>,
  pub reserve_tokens: Option<u64>,
  pub keep_recent_tokens: Option<u64>,
  #[serde(default)]
  pub no_compaction: bool,
  #[serde(default)]
  pub no_turn_summary: bool,
  #[serde(default)]
  pub no_vision: bool,
  #[serde(default)]
  pub no_session: bool,
  #[serde(default, deserialize_with = "path")]
  pub sessions_dir: Option<PathBuf>,
  #[serde(default, deserialize_with = "value_enum")]
  pub scrollbar: Option<ScrollbarMode>,
  #[serde(default)]
  pub no_bell: bool,
  #[serde(default, deserialize_with = "path")]
  pub mcp_config: Option<PathBuf>,
  #[serde(default)]
  pub no_mcp: bool,
  pub tools: Option<Vec<String>>,
  pub no_tools: Option<Vec<String>>,
}

/// A value named the way the flag takes it, so `provider = "openrouter"` is
/// written exactly as `--provider openrouter` is.
fn value_enum<'de, D: Deserializer<'de>, T: clap::ValueEnum>(de: D) -> Result<Option<T>, D::Error> {
  let Some(name) = Option::<String>::deserialize(de)? else {
    return Ok(None);
  };
  T::from_str(&name, false).map(Some).map_err(|_| {
    let names: Vec<String> = T::value_variants()
      .iter()
      .filter_map(|value| value.to_possible_value())
      .map(|value| value.get_name().to_string())
      .collect();
    serde::de::Error::custom(format!("no such value: {name} (one of {})", names.join(", ")))
  })
}

/// A path, with a `~` in front of it meaning what it would to the shell: no
/// shell reads this file to say so, and a path in a home directory is what it
/// is most likely to hold.
fn path<'de, D: Deserializer<'de>>(de: D) -> Result<Option<PathBuf>, D::Error> {
  let path = Option::<String>::deserialize(de)?;
  Ok(
    path.map(|path| match (path.strip_prefix('~'), std::env::var_os("HOME")) {
      (Some(rest), Some(home)) if rest.is_empty() || rest.starts_with('/') => {
        PathBuf::from(home).join(rest.trim_start_matches('/'))
      }
      _ => PathBuf::from(path),
    }),
  )
}

/// Where a file of this name is looked for, in the order they are read: the
/// system's first, the user's last, so the nearer file has the last word.
///
/// The search is the XDG one — `$XDG_CONFIG_HOME` then `$XDG_CONFIG_DIRS`,
/// each with their spec defaults — and nothing else. No dotfile in a home
/// directory, and none beside the project either: a file in the working
/// directory would be a file whose name depends on where fa was started.
pub fn files(name: &str, explicit: Option<&Path>) -> Vec<PathBuf> {
  if let Some(path) = explicit {
    return vec![path.to_path_buf()];
  }
  let mut files: Vec<PathBuf> = config_dirs()
    .into_iter()
    .rev()
    .map(|dir| dir.join("fa").join(name))
    .collect();
  files.dedup();
  files
}

/// The XDG configuration directories, nearest first.
fn config_dirs() -> Vec<PathBuf> {
  let var = |name| std::env::var_os(name).map(|value| value.to_string_lossy().into_owned());
  search(var("XDG_CONFIG_HOME"), var("HOME"), var("XDG_CONFIG_DIRS"))
}

/// The XDG search path, from the three variables that decide it, so what it
/// works out can be said without an environment to say it in.
///
/// Relative entries are dropped, which the specification asks for, and each
/// variable falls back to the default the specification gives it.
fn search(config_home: Option<String>, home: Option<String>, config_dirs: Option<String>) -> Vec<PathBuf> {
  let absolute = |dir: PathBuf| dir.is_absolute().then_some(dir);
  let first = config_home
    .map(PathBuf::from)
    .and_then(absolute)
    .or_else(|| home.map(|home| PathBuf::from(home).join(".config")));
  let rest = config_dirs.filter(|dirs| !dirs.is_empty()).unwrap_or("/etc/xdg".into());
  first
    .into_iter()
    .chain(rest.split(':').map(PathBuf::from).filter_map(absolute))
    .collect()
}

/// Read every file that is there, key by key, the nearer file winning.
///
/// Unlike `mcp.toml`, a file that makes no sense is a failure rather than a
/// note: it is read before the terminal is taken, where saying so costs
/// nothing, and a session started on half of what was meant — the wrong model,
/// the wrong endpoint — is worse than one not started. `strict` is for a file
/// asked for by name, where not being there is worth saying too.
pub fn load(paths: &[PathBuf], strict: bool) -> Result<Settings> {
  let mut merged = toml::Table::new();
  for path in paths {
    let text = match std::fs::read_to_string(path) {
      Ok(text) => text,
      Err(err) if err.kind() == std::io::ErrorKind::NotFound && !strict => continue,
      Err(err) => return Err(err).with_context(|| format!("could not read {}", path.display())),
    };
    // Each file on its own first, so a mistake is placed by the line it is on
    // in the file it is in, rather than somewhere in the merge of them all.
    toml::from_str::<Settings>(&text).with_context(|| format!("could not read {}", path.display()))?;
    let table: toml::Table = toml::from_str(&text)?;
    // The key and the command for it are one setting written two ways, so a
    // nearer file giving either replaces both.
    if table.contains_key("api-key") || table.contains_key("api-key-command") {
      merged.remove("api-key");
      merged.remove("api-key-command");
    }
    merged.extend(table);
  }
  let settings: Settings = toml::Value::Table(merged).try_into()?;
  if settings.api_key.is_some() && settings.api_key_command.is_some() {
    // Refused rather than resolved by a rule about which wins, as in
    // `mcp.toml`: one of the two is wrong, and quietly using either is how the
    // wrong one goes unnoticed.
    bail!("api-key is given both a value and a command");
  }
  Ok(settings)
}

/// What one line of shell prints, for a value a file should not hold.
pub async fn value(command: &str) -> Result<String> {
  let run = tokio::process::Command::new("bash")
    .arg("-c")
    .arg(command)
    // The terminal belongs to the transcript, so nothing here may ask it
    // anything: a pinentry that wanted this one would draw over the session
    // and wait for an answer no one can give it.
    .stdin(std::process::Stdio::null())
    .kill_on_drop(true)
    .output();
  let output = tokio::time::timeout(VALUE_TIMEOUT, run)
    .await
    .map_err(|_| anyhow::anyhow!("{command}: no answer in {}s", VALUE_TIMEOUT.as_secs()))?
    .with_context(|| format!("could not run {command}"))?;
  if !output.status.success() {
    // What it said for itself, never what it printed: the one is the reason
    // and the other is the secret it failed to produce.
    let said = String::from_utf8_lossy(&output.stderr);
    bail!("{command}: {}", said.lines().next().unwrap_or("said nothing"));
  }
  let value = String::from_utf8(output.stdout).with_context(|| format!("{command}: printed no text"))?;
  // A command prints a value with the newline it was printed with; the value
  // is the rest. Only the end, since a space at the front is a value that is
  // wrong rather than one that needs tidying.
  Ok(value.trim_end().to_string())
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn a_key_is_a_flag_name_and_a_value_is_written_as_the_flag_takes_it() {
    let read: Settings = toml::from_str(
      r#"
        provider = "openrouter"
        model = "m"
        max-tokens = 100
        scrollbar = "always"
        no-bell = true
        tools = ["read", "bash"]
        sessions-dir = "~/sessions"
      "#,
    )
    .expect("a readable file");
    assert_eq!(read.provider, Some(Provider::OpenRouter));
    assert_eq!(read.model.as_deref(), Some("m"));
    assert_eq!(read.max_tokens, Some(100));
    assert_eq!(read.scrollbar, Some(ScrollbarMode::Always));
    assert!(read.no_bell && !read.no_vision);
    assert_eq!(read.tools, Some(vec!["read".to_string(), "bash".to_string()]));
    if let Some(home) = std::env::var_os("HOME") {
      assert_eq!(read.sessions_dir, Some(PathBuf::from(home).join("sessions")));
    }
    assert_eq!(toml::from_str::<Settings>("").expect("empty"), Settings::default());

    let err = toml::from_str::<Settings>("modle = \"m\"")
      .expect_err("an unknown key is refused")
      .to_string();
    assert!(err.contains("modle"), "which key it was: {err}");
    let err = toml::from_str::<Settings>("provider = \"nobody\"")
      .expect_err("an unknown provider is refused")
      .to_string();
    assert!(
      err.contains("nobody") && err.contains("openrouter"),
      "and what it could be: {err}"
    );
  }

  #[test]
  fn the_nearest_file_has_the_last_word_key_by_key() {
    let dir = std::env::temp_dir().join(format!("fa-config-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("a directory");
    let system = dir.join("system.toml");
    let user = dir.join("user.toml");
    std::fs::write(
      &system,
      "model = \"system\"\nno-bell = true\napi-key-command = \"pass show k\"",
    )
    .expect("a file");
    std::fs::write(&user, "model = \"user\"\napi-key = \"k\"").expect("a file");

    let read = load(&[system.clone(), user.clone()], false).expect("both files");
    assert_eq!(read.model.as_deref(), Some("user"), "read last, so it wins");
    assert!(read.no_bell, "the rest of the other file stays");
    assert_eq!(read.api_key.as_deref(), Some("k"));
    assert_eq!(read.api_key_command, None, "a key written either way replaces both");

    // Given both ways in one place, it is wrong in one of them.
    let both = dir.join("both.toml");
    std::fs::write(&both, "api-key = \"k\"\napi-key-command = \"true\"").expect("a file");
    assert!(load(&[both], false).is_err());

    // A file that is not there is only worth saying when it was asked for.
    let missing = dir.join("nowhere.toml");
    assert_eq!(
      load(std::slice::from_ref(&missing), false).expect("nothing"),
      Settings::default()
    );
    assert!(load(&[missing], true).is_err());

    // One that makes no sense says where in it the trouble is.
    let broken = dir.join("broken.toml");
    std::fs::write(&broken, "model = \"m\"\nno-bell = maybe").expect("a file");
    let err = format!("{:#}", load(&[user, broken], false).expect_err("a broken file"));
    assert!(
      err.contains("broken.toml") && err.contains("line 2"),
      "where it went wrong: {err}"
    );
    let _ = std::fs::remove_dir_all(&dir);
  }

  #[tokio::test]
  async fn a_value_can_be_what_a_command_prints_and_what_went_wrong_is_never_that() {
    // What the command printed is the value, without the newline it was
    // printed with; a line of shell, not an argv.
    assert_eq!(value("printf 'secret \\n'").await.expect("a value"), "secret");
    assert_eq!(value("echo $((1 + 1))").await.expect("a value"), "2");
    // Nothing here may stop to ask the terminal anything, so a command that
    // reads is given the end of the input rather than the session's keys.
    assert_eq!(value("cat").await.expect("no waiting"), "");
    // A command that failed is told about as what it said for itself, never
    // as what it printed: that is the secret it half produced.
    let err = format!(
      "{:#}",
      value("echo $((111 * 111)); echo locked >&2; exit 1")
        .await
        .expect_err("a command that failed")
    );
    assert!(err.contains("locked"), "why: {err}");
    assert!(!err.contains("12321"), "but never the output: {err}");
  }

  #[test]
  fn the_search_path_is_the_xdg_one_and_only_that() {
    let path = |config_home: Option<&str>, home: Option<&str>, dirs: Option<&str>| {
      search(
        config_home.map(str::to_string),
        home.map(str::to_string),
        dirs.map(str::to_string),
      )
    };
    // Nearest first here; a relative entry is no entry at all.
    assert_eq!(
      path(
        Some("/home/someone/.config"),
        None,
        Some("/etc/xdg:relative/ignored:/opt/xdg")
      ),
      [
        PathBuf::from("/home/someone/.config"),
        PathBuf::from("/etc/xdg"),
        PathBuf::from("/opt/xdg"),
      ]
    );
    // Each variable falls back to what the specification says it means.
    assert_eq!(
      path(None, Some("/home/someone"), None),
      [PathBuf::from("/home/someone/.config"), PathBuf::from("/etc/xdg")]
    );
    assert_eq!(
      path(Some("relative"), Some("/home/someone"), None)[0],
      PathBuf::from("/home/someone/.config")
    );
    // Nowhere to look is not somewhere to look.
    assert_eq!(path(None, None, Some("")), [PathBuf::from("/etc/xdg")]);
  }

  #[test]
  fn the_files_are_read_system_first_and_never_from_the_working_directory() {
    let found = files(FILE, None);
    assert!(
      found.iter().all(|path| path.ends_with(Path::new("fa").join(FILE))),
      "only ever that one name under the search path: {found:?}"
    );
    assert!(
      !found.iter().any(|path| path.is_relative()),
      "nothing beside the project: {found:?}"
    );
    // The nearest directory is read last, so its keys win.
    let dirs = config_dirs();
    assert_eq!(found.len(), dirs.len(), "one file per directory: {found:?} {dirs:?}");
    if let (Some(nearest), Some(last)) = (dirs.first(), found.last()) {
      assert_eq!(last, &nearest.join("fa").join(FILE));
    }
    // Asked for by name, that is the only one.
    assert_eq!(
      files(FILE, Some(Path::new("/tmp/one.toml"))),
      [PathBuf::from("/tmp/one.toml")]
    );
  }
}
