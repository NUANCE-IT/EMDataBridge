"""Identity: ORCID validation, keys, signatures, trust and the NUScope key import."""
import json
import stat

import pytest

from EMDataBridge import identity as idn


@pytest.fixture(autouse=True)
def homes(tmp_path, monkeypatch):
    monkeypatch.setenv("EMDATABRIDGE_HOME", str(tmp_path / "emdb"))
    monkeypatch.setenv("NUSCOPE_HOME", str(tmp_path / "nuscope"))
    monkeypatch.delenv("EMDATABRIDGE_KEY_PASSPHRASE", raising=False)
    monkeypatch.delenv("NUSCOPE_KEY_PASSPHRASE", raising=False)
    return tmp_path


def test_orcid_validation():
    # ORCID's own documented example iD
    assert idn.normalize_orcid("0000-0002-1825-0097") == "https://orcid.org/0000-0002-1825-0097"
    assert idn.normalize_orcid("https://orcid.org/0000000218250097") == "https://orcid.org/0000-0002-1825-0097"
    assert idn.normalize_orcid("0000-0002-9079-593X") == "https://orcid.org/0000-0002-9079-593X"
    assert idn.normalize_orcid("") is None
    with pytest.raises(ValueError, match="check digit"):
        idn.normalize_orcid("0000-0002-1825-0098")
    with pytest.raises(ValueError, match="not an ORCID"):
        idn.normalize_orcid("orcid please")


def test_create_sign_verify(homes):
    me = idn.create("Ada Lovelace", orcid="0000-0002-1825-0097")
    assert idn.me() == me and me.orcid.endswith("0097")
    pem = idn.key_dir() / idn.KEY_FILE
    assert stat.S_IMODE(pem.stat().st_mode) == 0o600
    claims = {"seal": "ab" * 32}
    e = idn.sign(claims, role="author")
    c = idn.verify(claims, e)
    assert c.valid and c.trusted and c.signer["orcid"] == me.orcid
    assert not idn.verify({"seal": "cd" * 32}, e)                       # other claims
    forged = json.loads(json.dumps(e)); forged["signer"]["name"] = "Someone Else"
    assert not idn.verify(claims, forged)                                # edited signer record
    with pytest.raises(FileExistsError):
        idn.create("Ada Lovelace")


def test_bad_orcid_is_refused_at_creation():
    with pytest.raises(ValueError):
        idn.create("Ada Lovelace", orcid="0000-0002-1825-0098")
    assert idn.me() is None


def test_trust_export_untrust(homes, monkeypatch):
    idn.create("Ada Lovelace")
    out = idn.export(homes / "ada.json")
    e = idn.sign({"x": 1})
    monkeypatch.setenv("EMDATABRIDGE_HOME", str(homes / "other"))
    assert idn.verify({"x": 1}, e).valid and not idn.verify({"x": 1}, e).trusted
    ada = idn.trust(out)
    assert idn.verify({"x": 1}, e).trusted
    assert idn.untrust(ada.fingerprint) and not idn.verify({"x": 1}, e).trusted


def test_passphrase(homes, monkeypatch):
    idn.create("Ada Lovelace", passphrase="correct horse")
    with pytest.raises(TypeError):                                       # encrypted key, no passphrase
        idn.sign({"x": 1})
    monkeypatch.setenv("EMDATABRIDGE_KEY_PASSPHRASE", "correct horse")
    assert idn.verify({"x": 1}, idn.sign({"x": 1})).valid


def test_imports_a_nuscope_03_key(homes):
    """Keys made by NUScope 0.3 move to the shared identity, and old signatures stay valid."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ed25519
    import base64
    kd = homes / "nuscope" / "keys"; kd.mkdir(parents=True)
    sk = ed25519.Ed25519PrivateKey.generate()
    (kd / "nuscope_ed25519.pem").write_bytes(sk.private_bytes(serialization.Encoding.PEM,
                                                              serialization.PrivateFormat.PKCS8,
                                                              serialization.NoEncryption()))
    pub = base64.b64encode(sk.public_key().public_bytes(serialization.Encoding.Raw,
                                                        serialization.PublicFormat.Raw)).decode()
    info = {"name": "Ada Lovelace", "orcid": "https://orcid.org/0000-0002-1825-0097", "public_key": pub,
            "fingerprint": idn.fingerprint(pub), "created": "2026-10-13"}
    (kd / "nuscope_ed25519.pub.json").write_text(json.dumps(info))
    me = idn.me()
    assert me.fingerprint == info["fingerprint"] and (kd / "nuscope_ed25519.pem").exists()
    assert idn.verify({"seal": "s"}, idn.sign({"seal": "s"})).trusted


def test_cli_identity(homes, capsys, monkeypatch):
    from EMDataBridge.cli import main
    monkeypatch.setattr("sys.argv", ["emdb", "identity", "create", "Ada Lovelace", "--orcid", "0000-0002-1825-0097"])
    assert main() == 0
    monkeypatch.setattr("sys.argv", ["emdb", "identity", "show"])
    assert main() == 0
    assert "0000-0002-1825-0097" in capsys.readouterr().out


def test_light_import():
    """The identity layer must not pull in the format tools (pandas, sklearn, hyperspy...)."""
    import subprocess, sys
    code = ("import sys; from EMDataBridge import identity; "
            "bad = [m for m in ('pandas', 'sklearn', 'hyperspy', 'matplotlib', 'EMDataBridge.standardizer') if m in sys.modules]; "
            "print(bad); sys.exit(1 if bad else 0)")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
