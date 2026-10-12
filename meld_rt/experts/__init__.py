"""frozen experts: vad, sensevoice (asr + audio posteriors), text (roberta
embedding), prosody. each module imports its own runtime lazily so importing
one does not load the others."""
