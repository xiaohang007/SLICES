# Bundled GFN-FF Runtime Libraries

The portable x86_64 Linux GFN-FF runtime contains private copies of BLAS,
LAPACK, `libgfortran`, `libgomp`, and `libquadmath`. They are placed by
`auditwheel` under `upstream/python/gfnff.libs` and loaded through the relative
RPATH in `libgfnff.so.0.1.1`.

| Runtime | Build package provenance | License |
| --- | --- | --- |
| BLAS 3.9.0 | conda-forge `libblas-3.9.0-1_h86c2bf4_netlib` | BSD-3-Clause |
| LAPACK 3.9.0 | conda-forge `liblapack-3.9.0-14_h8876d29_netlib` | BSD-3-Clause |
| GNU Fortran runtime | conda-forge `libgfortran5-15.2.0-h68bc16d_19` | GPL-3.0-only WITH GCC-exception-3.1 |
| GNU OpenMP runtime | conda-forge `libgomp-15.2.0-he0feb66_19` | GPL-3.0-only WITH GCC-exception-3.1 |
| GNU Quadmath runtime | GCC runtime accompanying the GFortran toolchain | GPL-3.0-only WITH GCC-exception-3.1 |

GFN-FF itself remains LGPL-3.0-or-later; its source and license are retained in
this vendored directory. Keep this file and the existing GFN-FF license files
when redistributing the `src` bundle. Exact filenames and hashes are recorded
in `portable_manifest.json`.
