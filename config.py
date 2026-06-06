from __future__ import annotations
import os
from pathlib import Path

# ---------------- paths ----------------
#  base data dir; benchmarks cache; tools (JUnit JAR); results CSV target
DATA_DIR = Path(os.environ.get("CODEGEN_DATA_DIR", "/content/drive/MyDrive/codegen_task3_v4"))
BENCHMARKS_DIR = DATA_DIR / "benchmarks"          # 
TOOLS_DIR      = DATA_DIR / "tools"               #   (JUnit5 standalone JAR)
RESULTS_DIR    = DATA_DIR / "results"             #   (baselines.csv lives here)


#  JUnit5 console-launcher (downloaded once by setup_java_runtime, used by lib/execution.run_java)
JUNIT_JAR_VERSION = "1.10.1"
JUNIT_JAR = TOOLS_DIR / f"junit-platform-console-standalone-{JUNIT_JAR_VERSION}.jar"
JUNIT_JAR_URL = (
    f"https://repo1.maven.org/maven2/org/junit/platform/"
    f"junit-platform-console-standalone/{JUNIT_JAR_VERSION}/"
    f"junit-platform-console-standalone-{JUNIT_JAR_VERSION}.jar"
)
# ---------------- models ----------------

QWEN_1_5B = "Qwen/Qwen2.5-Coder-1.5B-Instruct"  
QWEN_7B   = "Qwen/Qwen2.5-Coder-7B-Instruct"  

MODELS = {"1.5b": QWEN_1_5B, "7b": QWEN_7B}     

# ---------------- run constants ----------------
SEED = int(os.environ.get("CODEGEN_SEED", "42"))                                  



#  decoding for K=0 baselines (greedy). DECODING_SAMPLED is for pass@10 (cp3 sampling).
DECODING_GREEDY  = {"do_sample": False, "temperature": 0.0, "max_new_tokens": 512}    

RETRIEVAL_TOKEN_CAP = 8000                                                        

# subprocess timeout for Python and Java sandboxes
EXEC_TIMEOUT_S = 10                                                               

#   CODEGEN_SMOKE=1   -> 20 problems per benchmark (pipeline smoke test)          
#   CODEGEN_SAMPLE=N  -> N problems per benchmark (reproducible random subset)    
#   neither           -> full benchmarks (164 / 257 / 164 / ...)                  
SMOKE = os.environ.get("CODEGEN_SMOKE", "0") == "1"                               
_SAMPLE_RAW = os.environ.get("CODEGEN_SAMPLE", "")
SAMPLE_SIZE = int(_SAMPLE_RAW) if _SAMPLE_RAW.isdigit() and int(_SAMPLE_RAW) > 0 else None  

BENCHMARK_LIMIT = 20 if SMOKE else SAMPLE_SIZE  # None = all problems             



def ensure_dirs() -> None:   
    for d in [DATA_DIR, BENCHMARKS_DIR, TOOLS_DIR, RESULTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
