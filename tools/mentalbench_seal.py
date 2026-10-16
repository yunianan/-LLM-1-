"""Windows CurrentUser DPAPI sealing for local evaluator-only material.

This protects stored plaintext, not access against the same Windows account.
Use a separate evaluator account for stronger operational separation.
"""
import ctypes
from ctypes import wintypes
import json
import os
import zlib


MAGIC = b"MENTALBENCH-EVAL-DPAPI-V2\x00"


class _Blob(ctypes.Structure):
    _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_ubyte))]


def _crypt(data, decrypt=False):
    if os.name != "nt":
        raise RuntimeError("DPAPI seal requires Windows; no plaintext fallback is allowed")
    buffer = ctypes.create_string_buffer(data)
    source = _Blob(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte)))
    target = _Blob()
    crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    kernel32.LocalFree.restype = ctypes.c_void_p
    if decrypt:
        function = crypt32.CryptUnprotectData
        function.argtypes = [ctypes.POINTER(_Blob), ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                             ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(_Blob)]
        second = None
    else:
        function = crypt32.CryptProtectData
        function.argtypes = [ctypes.POINTER(_Blob), wintypes.LPCWSTR, ctypes.c_void_p, ctypes.c_void_p,
                             ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(_Blob)]
        second = "MentalBench evaluator material v2"
    function.restype = wintypes.BOOL
    if not function(ctypes.byref(source), second, None, None, None, 1, ctypes.byref(target)):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        return ctypes.string_at(target.pbData, target.cbData)
    finally:
        kernel32.LocalFree(target.pbData)


def seal(payload):
    encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    return MAGIC + _crypt(zlib.compress(encoded))


def unseal(data):
    if not data.startswith(MAGIC):
        raise ValueError("Unrecognized sealed material")
    return json.loads(zlib.decompress(_crypt(data[len(MAGIC):], decrypt=True)).decode("utf-8"))
