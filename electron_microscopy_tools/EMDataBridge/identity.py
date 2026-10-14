"""Researcher identity for electron-microscopy data: ORCID, signing keys and trust.

One identity is shared by every tool that handles the data (NUScope, NUScope-TOMO,
NUANCE-MCP, EMDataBridge itself), so a dataset can say who acquired, processed
or approved it and prove it with a signature.

* :class:`Identity` is a person: name, ORCID iD (validated, including its check
  digit) and the fingerprint of their Ed25519 public key.
* ``create()`` makes your key. The private key is stored in
  ``~/.emdatabridge/identity`` (or ``$EMDATABRIDGE_HOME/identity``), readable only
  by you, and encrypted when ``EMDATABRIDGE_KEY_PASSPHRASE`` is set.
* ``sign(claims)`` signs a small JSON document (for example a file's seal) with
  your key; ``verify(claims, entry)`` checks a signature from anyone.
* ``trust()`` adds someone's public key to your trusted list; your own key is
  trusted automatically.

The module needs only the standard library and ``cryptography``.

    from EMDataBridge import identity
    me = identity.create("Roberto dos Reis", orcid="0000-0002-....")
    entry = identity.sign({"seal": "<sha256>"}, role="author")
    identity.verify({"seal": "<sha256>"}, entry).valid
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

KEY_FILE = "ed25519.pem"
PUB_FILE = "identity.json"
TRUST_FILE = "trusted_keys.json"
ALG = "Ed25519"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def home() -> Path:
    return Path(os.environ.get("EMDATABRIDGE_HOME", Path.home() / ".emdatabridge")).expanduser()


def key_dir() -> Path:
    return home() / "identity"


def _legacy_dirs() -> list[Path]:
    """Where NUScope 0.3 kept keys (nuscope_ed25519.pem), honouring NUSCOPE_HOME."""
    return [Path(os.environ.get("NUSCOPE_HOME", Path.home() / ".nuscope")).expanduser() / "keys"]


def canonical(obj: Any) -> bytes:
    """Deterministic JSON bytes (sorted keys, no whitespace, UTF-8): what gets hashed and signed."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=True,
                      default=str).encode()


# ------------------------------------------------------------------ ORCID
_ORCID = re.compile(r"(?:https?://orcid\.org/)?(\d{4})-?(\d{4})-?(\d{4})-?(\d{3}[\dX])$", re.I)


def orcid_checksum(base15: str) -> str:
    """ISO 7064 MOD 11-2 check digit of the first 15 digits of an ORCID iD."""
    total = 0
    for c in base15:
        total = (total + int(c)) * 2
    r = (12 - total % 11) % 11
    return "X" if r == 10 else str(r)


def normalize_orcid(value: Optional[str]) -> Optional[str]:
    """Return ``https://orcid.org/xxxx-xxxx-xxxx-xxxx`` or raise ValueError. Empty input gives None."""
    if not value or not str(value).strip():
        return None
    m = _ORCID.match(str(value).strip())
    if not m:
        raise ValueError(f"{value!r} is not an ORCID iD (expected 0000-0000-0000-0000 or https://orcid.org/...)")
    digits = "".join(m.groups()).upper()
    if orcid_checksum(digits[:15]) != digits[15]:
        raise ValueError(f"{value!r} is not a valid ORCID iD: the check digit does not match (typo?)")
    return "https://orcid.org/" + "-".join(digits[i:i + 4] for i in range(0, 16, 4))


# ------------------------------------------------------------------ identities
def fingerprint(public_key_b64: str) -> str:
    """Short key id: first 16 hex digits of SHA-256 of the raw public key, in groups of four."""
    h = hashlib.sha256(base64.b64decode(public_key_b64)).hexdigest()[:16]
    return ":".join(h[i:i + 4] for i in range(0, 16, 4))


@dataclass
class Identity:
    name: str
    orcid: Optional[str]
    public_key: str          # base64 raw Ed25519 public key
    fingerprint: str
    created: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Identity":
        ident = cls(**{k: d.get(k) for k in ("name", "orcid", "public_key", "fingerprint", "created")})
        if fingerprint(ident.public_key) != ident.fingerprint:
            raise ValueError("the fingerprint does not match the public key")
        return ident

    def as_prov_agent(self) -> dict:
        """W3C PROV / RO-Crate style person record."""
        return {"@id": self.orcid or f"#key-{self.fingerprint}", "@type": "Person", "name": self.name,
                **({"identifier": self.orcid} if self.orcid else {}), "keyFingerprint": self.fingerprint}

    def __str__(self) -> str:
        return f"{self.name}{f' ({self.orcid})' if self.orcid else ''} [{self.fingerprint}]"


def _crypto():
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import ed25519
    except ImportError as e:  # pragma: no cover
        raise ImportError("signing needs the 'cryptography' package: pip install cryptography") from e
    return serialization, ed25519


def _passphrase(passphrase: Optional[str]) -> Optional[bytes]:
    p = passphrase or os.environ.get("EMDATABRIDGE_KEY_PASSPHRASE") or os.environ.get("NUSCOPE_KEY_PASSPHRASE")
    return p.encode() if p else None


def _write_private(pem: bytes):
    d = key_dir()
    d.mkdir(parents=True, exist_ok=True)
    os.chmod(d, 0o700)
    fd = os.open(d / KEY_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(pem)


def create(name: str, orcid: Optional[str] = None, *, passphrase: Optional[str] = None,
           overwrite: bool = False) -> Identity:
    """Create your identity and Ed25519 signing key."""
    if not name or not name.strip():
        raise ValueError("enter your name")
    orcid = normalize_orcid(orcid)
    serialization, ed25519 = _crypto()
    if (key_dir() / KEY_FILE).exists() and not overwrite:
        raise FileExistsError(f"you already have a key in {key_dir()} (pass overwrite=True to replace it)")
    sk = ed25519.Ed25519PrivateKey.generate()
    pw = _passphrase(passphrase)
    enc = serialization.BestAvailableEncryption(pw) if pw else serialization.NoEncryption()
    _write_private(sk.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, enc))
    pub = base64.b64encode(sk.public_key().public_bytes(serialization.Encoding.Raw,
                                                        serialization.PublicFormat.Raw)).decode()
    ident = Identity(name=name.strip(), orcid=orcid, public_key=pub, fingerprint=fingerprint(pub), created=now())
    (key_dir() / PUB_FILE).write_text(json.dumps(ident.to_dict(), indent=2))
    trust(ident)
    return ident


def import_legacy() -> Optional[Identity]:
    """Copy a key made by NUScope 0.3 (``~/.nuscope/keys``) into the shared identity folder, once.

    The original files are left in place."""
    if (key_dir() / KEY_FILE).exists():
        return None
    for d in _legacy_dirs():
        pem, pub = d / "nuscope_ed25519.pem", d / "nuscope_ed25519.pub.json"
        if pem.exists() and pub.exists():
            info = json.loads(pub.read_text())
            ident = Identity.from_dict(info)
            _write_private(pem.read_bytes())
            (key_dir() / PUB_FILE).write_text(json.dumps(ident.to_dict(), indent=2))
            legacy_trust = d.parent / "trusted_keys.json"
            if legacy_trust.exists():
                for k in json.loads(legacy_trust.read_text()).values():
                    try:
                        trust(k)
                    except (ValueError, KeyError, TypeError):
                        pass
            trust(ident)
            return ident
    return None


def me() -> Optional[Identity]:
    """Your identity, or None if you have not created one."""
    if not (key_dir() / KEY_FILE).exists():
        import_legacy()
    p = key_dir() / PUB_FILE
    if not (p.exists() and (key_dir() / KEY_FILE).exists()):
        return None
    return Identity.from_dict(json.loads(p.read_text()))


def _private_key(passphrase: Optional[str] = None):
    serialization, _ = _crypto()
    return serialization.load_pem_private_key((key_dir() / KEY_FILE).read_bytes(), password=_passphrase(passphrase))


# ------------------------------------------------------------------ trust
def trusted() -> dict[str, dict]:
    p = home() / TRUST_FILE
    return json.loads(p.read_text()) if p.exists() else {}


def trust(key: Identity | dict | str | Path) -> Identity:
    """Trust a public key: an Identity, its dict, or a ``.json`` file someone shared with you."""
    if isinstance(key, (str, Path)):
        key = json.loads(Path(key).read_text())
    ident = key if isinstance(key, Identity) else Identity.from_dict(key)
    t = trusted()
    t[ident.fingerprint] = ident.to_dict() | {"trusted_at": now()}
    home().mkdir(parents=True, exist_ok=True)
    (home() / TRUST_FILE).write_text(json.dumps(t, indent=2))
    return ident


def untrust(fp: str) -> bool:
    """Remove a key from your trusted list. Returns False if it was not there."""
    t = trusted()
    if t.pop(fp, None) is None:
        return False
    (home() / TRUST_FILE).write_text(json.dumps(t, indent=2))
    return True


# ------------------------------------------------------------------ signing
@dataclass
class Check:
    valid: bool
    trusted: bool
    signer: dict

    def __bool__(self) -> bool:
        return self.valid


def _signed_bytes(claims: dict, signer: dict) -> bytes:
    if "signer" in claims:
        raise ValueError("'signer' is reserved in signed claims")
    return canonical({**claims, "signer": signer})


def sign(claims: dict, *, role: str = "author", passphrase: Optional[str] = None) -> dict:
    """Sign ``claims`` (a small JSON object) with your key.

    Returns ``{"signer": {...}, "signature": base64}``. The signature covers the
    claims and the signer record (name, ORCID, key, role and time)."""
    ident = me()
    if ident is None:
        raise FileNotFoundError("no identity yet; create one with: emdb identity create \"Your Name\" --orcid ...")
    signer = {"name": ident.name, "orcid": ident.orcid, "fingerprint": ident.fingerprint,
              "public_key": ident.public_key, "alg": ALG, "role": role, "signed_at": now()}
    sig = _private_key(passphrase).sign(_signed_bytes(claims, signer))
    return {"signer": signer, "signature": base64.b64encode(sig).decode()}


def verify(claims: dict, entry: dict) -> Check:
    """Check a signature entry made by :func:`sign` (by anyone) against the claims."""
    _, ed25519 = _crypto()
    signer = entry.get("signer", {})
    valid = False
    try:
        if signer.get("alg", ALG) == ALG and fingerprint(signer["public_key"]) == signer["fingerprint"]:
            pk = ed25519.Ed25519PublicKey.from_public_bytes(base64.b64decode(signer["public_key"]))
            pk.verify(base64.b64decode(entry["signature"]), _signed_bytes(claims, signer))
            valid = True
    except Exception:
        valid = False
    return Check(valid=valid, trusted=valid and signer.get("fingerprint") in trusted(), signer=signer)


def export(path: str | Path) -> Path:
    """Write your public identity (no private key) to share with collaborators."""
    ident = me()
    if ident is None:
        raise FileNotFoundError("no identity yet")
    path = Path(path).expanduser()
    path.write_text(json.dumps(ident.to_dict(), indent=2))
    return path


__all__ = ["Identity", "Check", "create", "me", "sign", "verify", "trust", "untrust", "trusted", "export",
           "normalize_orcid", "orcid_checksum", "fingerprint", "canonical", "home", "import_legacy"]
