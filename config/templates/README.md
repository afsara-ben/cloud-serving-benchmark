`llama-3.1-benchmark.jinja` is the exact `tokenizer.chat_template` from the existing
[8B Q4 GGUF](https://huggingface.co/mradermacher/Meta-Llama-3.1-8B-Instruct-i1-GGUF/blob/15a6b2be4570af9886c5cde164ddc8f41b4d7520/Meta-Llama-3.1-8B-Instruct.i1-Q4_K_M.gguf).
The source GGUF SHA256 is `85465a58bf902b8fe36e1e719fd1ddc23305110a6b19233f10560dfd1180f77a`.

All four 8B study formats explicitly use this template. The original FP16
checkpoint's embedded template adds knowledge/date metadata to ordinary system
messages (20 extra tokens in the checked request). The shared template preserves
the existing study's Llama 3 role formatting and makes input rendering comparable.
No comments or extra newlines were added to the Jinja text itself.
