# Preservation Audio Files

This directory should contain the audio files (`.wav`) referenced by
`../preservation_data.json`. The audio is excluded from this code release
because of its size (~3.3 GB, 4562 files).

When this code is merged into the parent repository that ships the editing
benchmark, the audio is already provided there — no extra action is needed.

## If you are running this code standalone

Obtain the preservation audio (4562 files) from the parent repository's
`Preservation_data/audio/` directory and place each file here so the layout
matches:

```
Preservation_data/
├── preservation_data.json
└── audio/
    ├── 000041.wav
    ├── 000078.wav
    └── ...
```

The filenames listed in `preservation_data.json` must resolve to existing
files in this folder. The editing scripts (`edit_af3.py`, `edit_qwen.py`,
`edit_desta.py`) silently skip any preservation entry whose audio file is
missing, so an empty folder will not crash the run but will disable audio
preservation.
