"""env.save_key: persist a provider key to this installation's .env AND the live
process env, so a freshly-entered key is effective without a restart (the
load_provider_env memo would otherwise ignore a mid-session .env edit).

Also covers the two hardening properties of that write:
  * the file is created owner-only where the platform has mode bits (security-04);
  * an INSTALLED (non-source) harness writes to a per-user config dir instead of
    into site-packages, where 'pip install -U' would delete the key and a system
    Python would refuse the write outright (packaging-06)."""
import os
import pathlib
import stat
import tempfile

from harness import env


def _swap_env_path(fake):
    old = env.ENV_PATH
    env.ENV_PATH = fake
    return old


def test_save_key_writes_file_and_sets_environ():
    with tempfile.TemporaryDirectory() as d:
        fake = pathlib.Path(d) / "ENV_MGMT" / ".env"     # parent must be created
        old_path = _swap_env_path(fake)
        prev = os.environ.get("ZORA_TEST_KEY")
        try:
            env.save_key("ZORA_TEST_KEY", "secret-123")
            assert os.environ["ZORA_TEST_KEY"] == "secret-123"      # live env set now
            assert "ZORA_TEST_KEY=secret-123" in fake.read_text(encoding="utf-8")
            # updating replaces the line in place -- no duplicate accumulation
            env.save_key("ZORA_TEST_KEY", "secret-456")
            text = fake.read_text(encoding="utf-8")
            assert text.count("ZORA_TEST_KEY=") == 1
            assert "ZORA_TEST_KEY=secret-456" in text
            assert os.environ["ZORA_TEST_KEY"] == "secret-456"
        finally:
            env.ENV_PATH = old_path
            if prev is None:
                os.environ.pop("ZORA_TEST_KEY", None)
            else:
                os.environ["ZORA_TEST_KEY"] = prev


def test_save_key_preserves_other_lines():
    with tempfile.TemporaryDirectory() as d:
        fake = pathlib.Path(d) / ".env"
        fake.write_text("# a comment\nOTHER_KEY=keepme\n", encoding="utf-8")
        old_path = _swap_env_path(fake)
        prev = os.environ.get("ZORA_NEW_KEY")
        try:
            env.save_key("ZORA_NEW_KEY", "v")
            text = fake.read_text(encoding="utf-8")
            assert "OTHER_KEY=keepme" in text        # untouched
            assert "# a comment" in text             # comments preserved
            assert "ZORA_NEW_KEY=v" in text
        finally:
            env.ENV_PATH = old_path
            if prev is None:
                os.environ.pop("ZORA_NEW_KEY", None)
            else:
                os.environ["ZORA_NEW_KEY"] = prev


def test_save_key_collapses_duplicate_lines():
    # a pre-existing duplicate of the key must not survive: env._parse (the reader)
    # is last-wins, so a stale TRAILING duplicate would silently revert the just-
    # saved value on the next launch. save_key must leave one authoritative line.
    with tempfile.TemporaryDirectory() as d:
        fake = pathlib.Path(d) / ".env"
        fake.write_text("ANTHROPIC_API_KEY=old_first\nOTHER=keep\n"
                        "ANTHROPIC_API_KEY=old_last\n", encoding="utf-8")
        old_path = _swap_env_path(fake)
        prev = os.environ.get("ANTHROPIC_API_KEY")
        try:
            env.save_key("ANTHROPIC_API_KEY", "brand_new")
            text = fake.read_text(encoding="utf-8")
            assert text.count("ANTHROPIC_API_KEY=") == 1        # single authoritative line
            assert "brand_new" in text
            assert "old_first" not in text and "old_last" not in text
            assert "OTHER=keep" in text                         # unrelated line kept
            # what a fresh launch would apply matches what we just saved
            assert env._parse(text)["ANTHROPIC_API_KEY"] == "brand_new"
        finally:
            env.ENV_PATH = old_path
            if prev is None:
                os.environ.pop("ANTHROPIC_API_KEY", None)
            else:
                os.environ["ANTHROPIC_API_KEY"] = prev


def test_save_key_rejects_empty_value_and_name():
    # never write / never touch os.environ for a blank value or name
    for bad in ("", "   "):
        try:
            env.save_key("ZORA_UNUSED", bad)
        except ValueError:
            continue
        raise AssertionError("an empty value must be rejected")
    assert "ZORA_UNUSED" not in os.environ
    try:
        env.save_key("", "x")
    except ValueError:
        pass
    else:
        raise AssertionError("an empty name must be rejected")


def test_save_key_rejects_multiline_values_and_invalid_names():
    for value in ("first\nINJECTED=second", "first\rsecond", "first\0second"):
        try:
            env.save_key("ZORA_UNUSED", value)
        except ValueError:
            pass
        else:
            raise AssertionError("a .env value must not inject another line")
    for name in ("BAD-NAME", "BAD=NAME", "1BAD", "BAD\nNAME"):
        try:
            env.save_key(name, "x")
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid env name {name!r} must be rejected")


# --- security-04: the key file is not world-readable ------------------------
def test_save_key_file_is_owner_only():
    with tempfile.TemporaryDirectory() as d:
        fake = pathlib.Path(d) / "cfg" / ".env"
        old_path = _swap_env_path(fake)
        prev = os.environ.get("ZORA_PERM_KEY")
        try:
            env.save_key("ZORA_PERM_KEY", "s3cret")
            mode = stat.S_IMODE(fake.stat().st_mode)
            if os.name == "nt":
                # Windows has no POSIX mode bits (os.chmod there only toggles the
                # read-only attribute), so protection comes from the inherited
                # ACL. Assert only what is meaningful: the file was written.
                assert fake.read_text(encoding="utf-8").strip() \
                    == "ZORA_PERM_KEY=s3cret"
            else:
                assert mode & 0o077 == 0, \
                    f"group/other must have no access, got {oct(mode)}"
                assert mode & 0o600 == 0o600
        finally:
            env.ENV_PATH = old_path
            if prev is None:
                os.environ.pop("ZORA_PERM_KEY", None)
            else:
                os.environ["ZORA_PERM_KEY"] = prev


def test_save_key_tightens_a_preexisting_loose_file():
    # an .env written by an older version (default 0644) must not stay loose
    with tempfile.TemporaryDirectory() as d:
        fake = pathlib.Path(d) / ".env"
        fake.write_text("OTHER=keep\n", encoding="utf-8")
        os.chmod(fake, 0o644)
        old_path = _swap_env_path(fake)
        prev = os.environ.get("ZORA_LOOSE_KEY")
        try:
            env.save_key("ZORA_LOOSE_KEY", "v")
            assert "OTHER=keep" in fake.read_text(encoding="utf-8")
            if os.name != "nt":
                assert stat.S_IMODE(fake.stat().st_mode) & 0o077 == 0
        finally:
            env.ENV_PATH = old_path
            if prev is None:
                os.environ.pop("ZORA_LOOSE_KEY", None)
            else:
                os.environ["ZORA_LOOSE_KEY"] = prev


# --- packaging-06: where the writable .env lives ----------------------------
def test_source_checkout_keeps_writing_the_vendored_env():
    """The user's existing workflow (running from the repo) must not move."""
    assert env.is_source_checkout(), \
        "these tests run from the source tree, so this must be True here"
    assert env.resolve_env_path() == env.VENDORED_ENV_PATH
    assert env.ENV_PATH == env.VENDORED_ENV_PATH
    assert env.VENDORED_ENV_PATH.parts[-3:] == ("_vendor", "ENV_MGMT", ".env")


def test_installed_harness_writes_to_a_per_user_config_dir():
    """Under a wheel install the package sits in site-packages: writing a
    credential there is wiped by 'pip install -U' and PermissionErrors on a
    system Python, so the key must go to a per-user config file instead."""
    real = env.is_source_checkout
    env.is_source_checkout = lambda: False
    saved = {k: os.environ.get(k) for k in ("APPDATA", "LOCALAPPDATA",
                                            "XDG_CONFIG_HOME")}
    try:
        with tempfile.TemporaryDirectory() as d:
            if os.name == "nt":
                os.environ["APPDATA"] = d
            else:
                os.environ["XDG_CONFIG_HOME"] = d
            path = env.resolve_env_path()
            assert path == pathlib.Path(d) / env.APP_DIR / ".env"
            assert path != env.VENDORED_ENV_PATH
            # and it is genuinely writable through the normal entry point
            old_path = _swap_env_path(path)
            prev = os.environ.get("ZORA_USERDIR_KEY")
            try:
                env.save_key("ZORA_USERDIR_KEY", "u")
                assert path.exists() and "ZORA_USERDIR_KEY=u" in \
                    path.read_text(encoding="utf-8")
            finally:
                env.ENV_PATH = old_path
                if prev is None:
                    os.environ.pop("ZORA_USERDIR_KEY", None)
                else:
                    os.environ["ZORA_USERDIR_KEY"] = prev
    finally:
        env.is_source_checkout = real
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
