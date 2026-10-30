# Public CC0 test radiographs

| File | Source | Licence | Size | SHA-256 |
|---|---|---|---|---|
| `pa_cc0_2010.png` | https://commons.wikimedia.org/wiki/File:Chest_Xray_PA_3-8-2010.png | CC0 (public domain dedication), uploader "Stillwaterising" | 2412×1956 RGBA, 4,198,015 bytes | `a43431b089a6dfab031c10eceef6c821e4f326913bf9015f73f4a1f0446d2be1` |
| `lateral_cc0_2010.png` | https://commons.wikimedia.org/wiki/File:Chest_Xray_Lateral_3-8-2010.png | CC0, uploader "Stillwaterising" | 1956×2412 RGBA, 3,934,746 bytes | `d07a306fb285a8456739e0349fb208b64ade12eb4bb84bc09889c7b3ec7f99a7` |

- Licence verified through the Wikimedia API (`extmetadata.LicenseShortName = CC0`) on 2026-10-30.
- The two files appear to be the **same subject on the same date** (same uploader, same date in the title, same burned-in technologist marker). This was not otherwise verified.
- No CT exists for this subject.
- The images carry a laterality marker ("L") and technologist initials. They contain no patient identifiers.
- The images are git-ignored by policy (`validation_data/public_cc0/*.png`). The notebooks download them and verify these checksums.
