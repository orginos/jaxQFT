"""Readers for SciDAC/ILDG LIME files used by Chroma/QIO."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import re
import struct
from pathlib import Path
from typing import Iterator

import numpy as np

LIME_MAGIC = 0x456789AB
LIME_HEADER_BYTES = 144


@dataclass(frozen=True)
class LimeRecord:
    """LIME record header."""

    index: int
    offset: int
    version: int
    flags: int
    size: int
    type_name: str

    @property
    def data_offset(self) -> int:
        return self.offset + LIME_HEADER_BYTES

    @property
    def is_begin_message(self) -> bool:
        return bool(self.flags & 0x8000)

    @property
    def is_end_message(self) -> bool:
        return bool(self.flags & 0x4000)


@dataclass(frozen=True)
class ScidacFieldMeta:
    """Metadata for one SciDAC field record."""

    datatype: str
    precision: str
    colors: int
    spins: int
    typesize: int
    datacount: int
    dims: tuple[int, ...]
    recordtype: int | None
    private_record_index: int
    binary_record_index: int
    binary_record_type: str
    record_xml_index: int | None
    file_xml_index: int | None


def _normalize_path(path: str | Path) -> Path:
    return path if isinstance(path, Path) else Path(path)


def _read_header(block: bytes) -> tuple[int, int, int, int]:
    return struct.unpack(">IHHQ", block[:16])


def iter_lime_records(path: str | Path) -> Iterator[LimeRecord]:
    """Yield LIME headers from a file."""
    p = _normalize_path(path)
    with p.open("rb") as f:
        offset = 0
        idx = 0
        while True:
            hdr = f.read(LIME_HEADER_BYTES)
            if not hdr:
                return
            if len(hdr) < LIME_HEADER_BYTES:
                raise ValueError(f"Truncated LIME header at offset {offset} in {p}")
            magic, ver, flags, nbytes = _read_header(hdr)
            if magic != LIME_MAGIC:
                raise ValueError(
                    f"Bad LIME magic {magic:#x} at offset {offset} in {p} "
                    f"(expected {LIME_MAGIC:#x})"
                )
            type_name = hdr[16:144].split(b"\x00", 1)[0].decode("ascii", "replace")
            rec = LimeRecord(
                index=idx,
                offset=offset,
                version=int(ver),
                flags=int(flags),
                size=int(nbytes),
                type_name=type_name,
            )
            yield rec
            f.seek(rec.size, 1)
            pad = (-rec.size) & 7
            if pad:
                f.seek(pad, 1)
            offset = f.tell()
            idx += 1


def list_lime_records(path: str | Path) -> list[LimeRecord]:
    return list(iter_lime_records(path))


def read_lime_record_data(path: str | Path, rec: LimeRecord | int) -> bytes:
    """Read payload bytes for a record header or record index."""
    p = _normalize_path(path)
    rec_obj: LimeRecord
    if isinstance(rec, int):
        headers = list_lime_records(p)
        if rec < 0 or rec >= len(headers):
            raise IndexError(f"Record index {rec} out of range [0,{len(headers)})")
        rec_obj = headers[rec]
    else:
        rec_obj = rec
    with p.open("rb") as f:
        f.seek(rec_obj.data_offset)
        return f.read(rec_obj.size)


def _xml_tag_text(xml: str, tag: str) -> str | None:
    m = re.search(rf"<{tag}(?:\s[^>]*)?>(.*?)</{tag}>", xml, re.S)
    if m is None:
        return None
    return m.group(1).strip()


def _parse_int(text: str | None, default: int | None = None) -> int | None:
    if text is None:
        return default
    try:
        return int(text)
    except Exception:
        return default


def _parse_dims(scidac_file_xml: str | None, ildg_xml: str | None) -> tuple[int, ...]:
    dims: tuple[int, ...] = ()
    if scidac_file_xml:
        d = _xml_tag_text(scidac_file_xml, "dims")
        if d:
            vals = [int(x) for x in d.split() if x.strip()]
            if vals:
                dims = tuple(vals)
    if dims:
        return dims
    if ildg_xml:
        vals: list[int] = []
        for tag in ("lx", "ly", "lz", "lt"):
            txt = _xml_tag_text(ildg_xml, tag)
            if txt is None:
                break
            vals.append(int(txt))
        if vals:
            return tuple(vals)
    return ()


def find_scidac_fields(path: str | Path) -> list[ScidacFieldMeta]:
    """Find SciDAC field descriptors and linked binary payload records."""
    headers = list_lime_records(path)
    payloads: dict[int, bytes] = {}
    for rec in headers:
        if "xml" in rec.type_name or rec.type_name == "ildg-format":
            payloads[rec.index] = read_lime_record_data(path, rec)

    scidac_file_xml_idx = next((r.index for r in headers if r.type_name == "scidac-private-file-xml"), None)
    scidac_file_xml = None if scidac_file_xml_idx is None else payloads[scidac_file_xml_idx].decode("utf-8", "replace")

    ildg_fmt_idx = next((r.index for r in headers if r.type_name == "ildg-format"), None)
    ildg_fmt_xml = None if ildg_fmt_idx is None else payloads[ildg_fmt_idx].decode("utf-8", "replace")

    dims = _parse_dims(scidac_file_xml, ildg_fmt_xml)

    out: list[ScidacFieldMeta] = []
    for i, rec in enumerate(headers):
        if rec.type_name != "scidac-private-record-xml":
            continue
        xml = payloads.get(rec.index, b"").decode("utf-8", "replace")
        datatype = _xml_tag_text(xml, "datatype") or ""
        precision = _xml_tag_text(xml, "precision") or "F"
        colors = _parse_int(_xml_tag_text(xml, "colors"), 0) or 0
        spins = _parse_int(_xml_tag_text(xml, "spins"), 0) or 0
        typesize = _parse_int(_xml_tag_text(xml, "typesize"), 0) or 0
        datacount = _parse_int(_xml_tag_text(xml, "datacount"), 1) or 1
        recordtype = _parse_int(_xml_tag_text(xml, "recordtype"), None)

        record_xml_index = None
        if i + 1 < len(headers) and headers[i + 1].type_name == "scidac-record-xml":
            record_xml_index = headers[i + 1].index

        binary_record_index = None
        binary_record_type = None
        for j in range(i + 1, len(headers)):
            t = headers[j].type_name
            if t in ("ildg-binary-data", "scidac-binary-data"):
                binary_record_index = headers[j].index
                binary_record_type = t
                break
        if binary_record_index is None or binary_record_type is None:
            continue

        out.append(
            ScidacFieldMeta(
                datatype=datatype,
                precision=precision,
                colors=int(colors),
                spins=int(spins),
                typesize=int(typesize),
                datacount=int(datacount),
                dims=dims,
                recordtype=recordtype,
                private_record_index=rec.index,
                binary_record_index=binary_record_index,
                binary_record_type=binary_record_type,
                record_xml_index=record_xml_index,
                file_xml_index=scidac_file_xml_idx,
            )
        )
    return out


def _float_dtype_for_precision(precision: str) -> np.dtype:
    p = precision.upper().strip()
    if p == "F":
        return np.dtype(">f4")
    if p == "D":
        return np.dtype(">f8")
    raise ValueError(f"Unsupported SciDAC precision {precision!r}")


def _lexi_sites_to_lattice(data: np.ndarray, dims: tuple[int, ...]) -> np.ndarray:
    if not dims:
        return data
    nd = len(dims)
    if data.shape[0] != int(np.prod(dims)):
        raise ValueError(f"Site count mismatch: data has {data.shape[0]} sites, dims={dims}")
    reshaped = data.reshape(tuple(reversed(dims)) + data.shape[1:], order="C")
    axes = tuple(range(nd - 1, -1, -1)) + tuple(range(nd, reshaped.ndim))
    return np.transpose(reshaped, axes)


def decode_scidac_field(path: str | Path, field_index: int = 0) -> dict:
    """Decode one SciDAC field payload to complex NumPy arrays.

    Returns a dict with:
    - meta: ScidacFieldMeta
    - site_data: site-major array (V, datacount, ...)
    - lattice_data: lattice-axes array (*dims, datacount, ...)
    """
    fields = find_scidac_fields(path)
    if field_index < 0 or field_index >= len(fields):
        raise IndexError(f"field_index={field_index} out of range [0,{len(fields)})")
    meta = fields[field_index]
    raw = read_lime_record_data(path, meta.binary_record_index)

    fdt = _float_dtype_for_precision(meta.precision)
    scalars_per_word = int(fdt.itemsize)
    n_real_per_item = meta.typesize // scalars_per_word
    if meta.typesize % scalars_per_word != 0:
        raise ValueError(
            f"Incompatible typesize={meta.typesize} for precision={meta.precision} ({scalars_per_word} bytes)"
        )
    if n_real_per_item % 2 != 0:
        raise ValueError(f"Expected even re/im scalar count per site item, got {n_real_per_item}")

    reals = np.frombuffer(raw, dtype=fdt)
    reals = reals.astype(np.float32 if fdt.itemsize == 4 else np.float64, copy=False)

    volume = int(np.prod(meta.dims)) if meta.dims else None
    if volume is None:
        if meta.datacount * n_real_per_item == 0:
            raise ValueError("Missing dimensions and invalid record shape metadata")
        if reals.size % (meta.datacount * n_real_per_item) != 0:
            raise ValueError("Cannot infer volume from payload size and metadata")
        volume = reals.size // (meta.datacount * n_real_per_item)
    expected = volume * meta.datacount * n_real_per_item
    if reals.size != expected:
        raise ValueError(
            f"Binary length mismatch: got {reals.size} float words, expected {expected} "
            f"(V={volume}, datacount={meta.datacount}, n_real_per_item={n_real_per_item})"
        )

    site_real = reals.reshape(volume, meta.datacount, n_real_per_item)
    site_complex = site_real[..., 0::2] + 1j * site_real[..., 1::2]

    datatype = meta.datatype
    if "ColorMatrix" in datatype:
        n = meta.colors
        if n <= 0:
            raise ValueError(f"Invalid colors={meta.colors} in {datatype}")
        if site_complex.shape[-1] != n * n:
            raise ValueError(
                f"ColorMatrix payload mismatch: got {site_complex.shape[-1]} complex values, expected {n*n}"
            )
        site_data = site_complex.reshape(volume, meta.datacount, n, n)
    elif "DiracFermion" in datatype:
        ns = meta.spins
        nc = meta.colors
        if ns <= 0 or nc <= 0:
            raise ValueError(f"Invalid spins/colors ({meta.spins},{meta.colors}) in {datatype}")
        if site_complex.shape[-1] != ns * nc:
            raise ValueError(
                f"DiracFermion payload mismatch: got {site_complex.shape[-1]} complex values, expected {ns*nc}"
            )
        site_data = site_complex.reshape(volume, meta.datacount, ns, nc)
    elif "ColorVector" in datatype:
        nc = meta.colors
        if nc <= 0:
            raise ValueError(f"Invalid colors={meta.colors} in {datatype}")
        if site_complex.shape[-1] != nc:
            raise ValueError(
                f"ColorVector payload mismatch: got {site_complex.shape[-1]} complex values, expected {nc}"
            )
        site_data = site_complex.reshape(volume, meta.datacount, nc)
    elif (meta.spins > 0) and (meta.colors > 0) and (site_complex.shape[-1] == meta.spins * meta.colors):
        # Some QIO records use generic datatype "Lattice" with explicit spin/color metadata.
        site_data = site_complex.reshape(volume, meta.datacount, meta.spins, meta.colors)
    else:
        site_data = site_complex

    lattice_data = _lexi_sites_to_lattice(site_data, meta.dims)
    return {
        "meta": meta,
        "site_data": site_data,
        "lattice_data": lattice_data,
    }


def decode_scidac_gauge(path: str | Path, field_index: int = 0, batch_size: int = 1) -> np.ndarray:
    """Decode a gauge field to jaxQFT BM...IJ layout.

    Returns shape: (batch, mu, *dims, Nc, Nc)
    """
    dec = decode_scidac_field(path, field_index=field_index)
    meta: ScidacFieldMeta = dec["meta"]
    lat = dec["lattice_data"]
    if "ColorMatrix" not in meta.datatype:
        raise ValueError(f"Expected ColorMatrix field, got {meta.datatype}")
    if meta.datacount <= 0:
        raise ValueError(f"Expected datacount>0 for gauge links, got {meta.datacount}")
    # lattice_data is (*dims, datacount, Nc, Nc). Move mu/datacount in front.
    mu_axis = len(meta.dims)
    links = np.moveaxis(lat, mu_axis, 0)  # (mu, *dims, Nc, Nc)
    links = links[None, ...]  # (1, mu, *dims, Nc, Nc)
    if int(batch_size) != 1:
        links = np.repeat(links, int(batch_size), axis=0)
    return links


def decode_scidac_momentum(
    path: str | Path,
    field_index: int = 0,
    batch_size: int = 1,
    *,
    check_traceless_antihermitian: bool = True,
) -> tuple[np.ndarray, dict]:
    """Decode an SMD/HMC momentum file in BM...IJ layout.

    Momentum files produced by Chroma SMD checkpoints in your workflow are
    written as `QDP_F3_ColorMatrix` with `datacount=4`, same packing as gauge.
    This helper returns the matrix field and optional diagnostics.
    """
    mom = decode_scidac_gauge(path, field_index=field_index, batch_size=batch_size)
    info = {}
    if check_traceless_antihermitian:
        m = mom
        md = np.swapaxes(np.conjugate(m), -1, -2)
        denom = float(np.linalg.norm(m))
        rel_antiherm = float(np.linalg.norm(m + md) / denom) if denom > 0 else 0.0
        tr = np.trace(m, axis1=-2, axis2=-1)
        info = {
            "rel_antihermitian_error": rel_antiherm,
            "mean_abs_trace": float(np.mean(np.abs(tr))),
            "max_abs_trace": float(np.max(np.abs(tr))),
        }
    return mom, info


def decode_scidac_pseudofermion(
    path: str | Path,
    field_index: int | None = None,
    batch_size: int = 1,
) -> tuple[np.ndarray, ScidacFieldMeta]:
    """Decode pseudofermion field from SciDAC LIME.

    If `field_index` is None, auto-select the first likely pseudofermion leaf:
    recordtype=0, precision in {F,D}, spins>0, colors>0.
    """
    fields = find_scidac_fields(path)
    idx = field_index
    if idx is None:
        cand = [
            i
            for i, m in enumerate(fields)
            if m.precision.upper() in ("F", "D")
            and (m.recordtype in (0, None))
            and m.spins > 0
            and m.colors > 0
        ]
        if not cand:
            raise ValueError(
                "Could not auto-locate pseudofermion leaf field; pass --field-index."
            )
        idx = cand[0]

    dec = decode_scidac_field(path, int(idx))
    meta: ScidacFieldMeta = dec["meta"]
    lat = dec["lattice_data"]
    # expected shape: (*dims, datacount, spins, colors)
    if meta.datacount == 1:
        pf = lat[..., 0, :, :]
    else:
        pf = lat
    pf = pf[None, ...]  # batch axis
    if int(batch_size) != 1:
        pf = np.repeat(pf, int(batch_size), axis=0)
    return pf, meta


def _cli():
    ap = argparse.ArgumentParser(description="Inspect/decode SciDAC/ILDG LIME files.")
    ap.add_argument("file", type=str, help="path to .lime file")
    ap.add_argument("--field-index", type=int, default=0, help="SciDAC field index (default: 0)")
    ap.add_argument("--dump-xml", action=argparse.BooleanOptionalAction, default=False, help="print XML records")
    ap.add_argument("--decode", action=argparse.BooleanOptionalAction, default=False, help="decode selected field")
    ap.add_argument("--decode-gauge", action=argparse.BooleanOptionalAction, default=False, help="decode selected field as gauge and print shape")
    ap.add_argument("--decode-momentum", action=argparse.BooleanOptionalAction, default=False, help="decode selected field as momentum and print anti-H/traceless checks")
    ap.add_argument("--decode-pf", action=argparse.BooleanOptionalAction, default=False, help="decode pseudofermion leaf field (auto by default)")
    ap.add_argument("--check-plaq", action=argparse.BooleanOptionalAction, default=False, help="compute plaquette from decoded gauge")
    ap.add_argument("--beta", type=float, default=1.0, help="beta for temporary gauge-theory object in --check-plaq")
    ap.add_argument("--save-npz", type=str, default="", help="save decoded field arrays to NPZ")
    args = ap.parse_args()

    path = Path(args.file)
    recs = list_lime_records(path)
    print(f"File: {path}")
    print(f"Records: {len(recs)}")
    for r in recs:
        print(
            f"  idx={r.index:2d} off={r.offset:8d} size={r.size:8d} "
            f"flags=0x{r.flags:04x} type={r.type_name}"
        )
    fields = find_scidac_fields(path)
    print(f"SciDAC fields: {len(fields)}")
    for i, f in enumerate(fields):
        print(
            f"  field[{i}] datatype={f.datatype} precision={f.precision} "
            f"dims={f.dims} datacount={f.datacount} typesize={f.typesize} recordtype={f.recordtype} "
            f"private={f.private_record_index} binary={f.binary_record_index}:{f.binary_record_type}"
        )

    if args.dump_xml:
        for r in recs:
            if "xml" in r.type_name or r.type_name == "ildg-format":
                data = read_lime_record_data(path, r)
                txt = data.decode("utf-8", "replace")
                print(f"\n--- record[{r.index}] {r.type_name} ---\n{txt}")

    decoded = None
    if args.decode or args.decode_gauge or args.decode_momentum or args.check_plaq or args.save_npz:
        decoded = decode_scidac_field(path, field_index=int(args.field_index))
        meta: ScidacFieldMeta = decoded["meta"]
        print("Decoded field:")
        print(f"  datatype: {meta.datatype}")
        print(f"  dims: {meta.dims}")
        print(f"  site_data shape: {decoded['site_data'].shape}")
        print(f"  lattice_data shape: {decoded['lattice_data'].shape}")

    if args.decode_gauge or args.check_plaq:
        gauge = decode_scidac_gauge(path, field_index=int(args.field_index), batch_size=1)
        print(f"Gauge BM...IJ shape: {gauge.shape}")
        if args.check_plaq:
            import jax
            import jax.numpy as jnp
            from jaxqft.models.su3_ym import SU3YangMills

            dims = tuple(int(v) for v in decoded["meta"].dims)
            if len(dims) == 0:
                raise ValueError("Cannot run plaquette check without lattice dimensions")
            th = SU3YangMills(lattice_shape=dims, beta=float(args.beta), batch_size=1, layout="BMXYIJ", exp_method="su3")
            q = jax.device_put(jnp.asarray(gauge))
            plaq = np.asarray(th.average_plaquette(q))
            act = np.asarray(th.action(q))
            print(f"Plaquette(mean): {float(plaq.mean()):.12f}")
            print(f"Gauge action(mean): {float(act.mean()):.12f}")

    if args.decode_momentum:
        mom, info = decode_scidac_momentum(path, field_index=int(args.field_index), batch_size=1)
        print(f"Momentum BM...IJ shape: {mom.shape}")
        if info:
            print(
                "Momentum quality:"
                f" rel_antihermitian_error={info['rel_antihermitian_error']:.6e}"
                f" mean_abs_trace={info['mean_abs_trace']:.6e}"
                f" max_abs_trace={info['max_abs_trace']:.6e}"
            )

    if args.decode_pf:
        pf_idx = int(args.field_index)
        if pf_idx < 0:
            use_idx = None
        elif pf_idx < len(fields) and fields[pf_idx].precision.upper() not in ("F", "D"):
            use_idx = None
        else:
            use_idx = pf_idx
        pf, meta = decode_scidac_pseudofermion(path, field_index=use_idx, batch_size=1)
        print(
            f"Pseudofermion shape: {pf.shape}"
            f" (datatype={meta.datatype}, precision={meta.precision}, spins={meta.spins}, colors={meta.colors}, datacount={meta.datacount}, recordtype={meta.recordtype})"
        )
        print(
            "Pseudofermion norms:"
            f" ||pf||={float(np.linalg.norm(pf)):.6e}"
            f" mean|pf|={float(np.mean(np.abs(pf))):.6e}"
        )

    if args.save_npz:
        if decoded is None:
            decoded = decode_scidac_field(path, field_index=int(args.field_index))
        meta: ScidacFieldMeta = decoded["meta"]
        np.savez_compressed(
            args.save_npz,
            site_data=decoded["site_data"],
            lattice_data=decoded["lattice_data"],
            dims=np.asarray(meta.dims, dtype=np.int64),
            datatype=np.asarray(meta.datatype),
            precision=np.asarray(meta.precision),
            colors=np.asarray(meta.colors),
            spins=np.asarray(meta.spins),
            typesize=np.asarray(meta.typesize),
            datacount=np.asarray(meta.datacount),
        )
        print(f"Saved {args.save_npz}")


if __name__ == "__main__":
    _cli()
