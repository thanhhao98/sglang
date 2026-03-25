export MODEL=deepseek-ai/DeepSeek-V2
export MEM_FRAC=0.85
export BACKEND=fa3
export CONTEXT_LENGTH=163840
export MAX_CC=512
SGLANG_DCP=8 SGLANG_DCP_SYMM_ONLY=true NCCL_DEBUG=WARN PYTHONUNBUFFERED=1 TORCHINDUCTOR_FX_GRAPH_CACHE=1 \
	TORCHINDUCTOR_AUTOGRAD_CACHE=1 SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK=1 TORCHINDUCTOR_CACHE_DIR=/home/admin/inductor_root_cache \
	python3 -m sglang.launch_server   --model-path $MODEL --host 0.0.0.0   --port 8188   --trust-remote-code   --enable-cache-report   \
	--log-level info   --tp-size 8   --max-running-requests $MAX_CC --mem-fraction-static $MEM_FRAC   --chunked-prefill-size 32768   \
	--context-length $CONTEXT_LENGTH --attention-backend $BACKEND --disable-radix-cache   --enable-symm-mem --dcp-comm-backend a2a
