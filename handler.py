import runpod
from runpod.serverless.utils import rp_upload
import os
import websocket
import base64
import hashlib
import json
import uuid
import logging
import urllib.request
import urllib.parse
import binascii # Base64 에러 처리를 위해 import
import subprocess
import time


# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CUDA 검사 및 설정
def check_cuda_availability():
    """CUDA 사용 가능 여부를 확인하고 환경 변수를 설정합니다."""
    try:
        import torch
        if torch.cuda.is_available():
            logger.info("✅ CUDA is available and working")
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            return True
        else:
            logger.error("❌ CUDA is not available")
            raise RuntimeError("CUDA is required but not available")
    except Exception as e:
        logger.error(f"❌ CUDA check failed: {e}")
        raise RuntimeError(f"CUDA initialization failed: {e}")

# CUDA 검사 실행
try:
    cuda_available = check_cuda_availability()
    if not cuda_available:
        raise RuntimeError("CUDA is not available")
except Exception as e:
    logger.error(f"Fatal error: {e}")
    logger.error("Exiting due to CUDA requirements not met")
    exit(1)



server_address = os.getenv('SERVER_ADDRESS', '127.0.0.1')
client_id = str(uuid.uuid4())
def save_data_if_base64(data_input, temp_dir, output_filename):
    """
    입력 데이터가 Base64 문자열인지 확인하고, 맞다면 파일로 저장 후 경로를 반환합니다.
    만약 일반 경로 문자열이라면 그대로 반환합니다.
    """
    # 입력값이 문자열이 아니면 그대로 반환
    if not isinstance(data_input, str):
        return data_input

    try:
        # Base64 문자열은 디코딩을 시도하면 성공합니다.
        decoded_data = base64.b64decode(data_input)
        
        # 디렉토리가 존재하지 않으면 생성
        os.makedirs(temp_dir, exist_ok=True)
        
        # 디코딩에 성공하면, 임시 파일로 저장합니다.
        file_path = os.path.abspath(os.path.join(temp_dir, output_filename))
        with open(file_path, 'wb') as f: # 바이너리 쓰기 모드('wb')로 저장
            f.write(decoded_data)
        
        # 저장된 파일의 경로를 반환합니다.
        print(f"✅ Base64 입력을 '{file_path}' 파일로 저장했습니다.")
        return file_path

    except (binascii.Error, ValueError):
        # 디코딩에 실패하면, 일반 경로로 간주하고 원래 값을 그대로 반환합니다.
        print(f"➡️ '{data_input}'은(는) 파일 경로로 처리합니다.")
        return data_input
    
def queue_prompt(prompt):
    url = f"http://{server_address}:8188/prompt"
    logger.info(f"Queueing prompt to: {url}")
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode('utf-8')
    req = urllib.request.Request(url, data=data)
    return json.loads(urllib.request.urlopen(req).read())

def get_image(filename, subfolder, folder_type):
    url = f"http://{server_address}:8188/view"
    logger.info(f"Getting image from: {url}")
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    with urllib.request.urlopen(f"{url}?{url_values}") as response:
        return response.read()

def get_history(prompt_id):
    url = f"http://{server_address}:8188/history/{prompt_id}"
    logger.info(f"Getting history from: {url}")
    with urllib.request.urlopen(url) as response:
        return json.loads(response.read())

def get_images(ws, prompt):
    prompt_id = queue_prompt(prompt)['prompt_id']
    output_images = {}
    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message['type'] == 'executing':
                data = message['data']
                if data['node'] is None and data['prompt_id'] == prompt_id:
                    break
        else:
            continue

    history = get_history(prompt_id)[prompt_id]
    for node_id in history['outputs']:
        node_output = history['outputs'][node_id]
        images_output = []
        if 'images' in node_output:
            for image in node_output['images']:
                image_data = get_image(image['filename'], image['subfolder'], image['type'])
                # bytes 객체를 base64로 인코딩하여 JSON 직렬화 가능하게 변환
                if isinstance(image_data, bytes):
                    import base64
                    image_data = base64.b64encode(image_data).decode('utf-8')
                images_output.append(image_data)
        output_images[node_id] = images_output

    return output_images

def load_workflow(workflow_path):
    with open(workflow_path, 'r') as file:
        return json.load(file)

# 새 워크플로우 파일명: 이미지 개수별
_WORKFLOW_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "workflow")
_WORKFLOW_FILES = {
    1: "qwen_image_edit_1_1image.json",
    2: "qwen_image_edit_1_2image.json",
    3: "qwen_image_edit_1_3image.json",
}

# 워크플로우별 노드 ID (이미지 개수에 따라 사용)
# 1-image: LoadImage=78, KSampler(seed)=3, prompt=111
# 2-image: 위 + LoadImage2=117
# 3-image: 위 + LoadImage3=119
_NODE_IMAGE_1 = "78"
_NODE_IMAGE_2 = "117"
_NODE_IMAGE_3 = "119"
_NODE_SEED = "3"
_NODE_PROMPT = "111"
_NODE_WIDTH = "128"   # 현재 워크플로우에는 없음(선택 적용)
_NODE_HEIGHT = "129"  # 현재 워크플로우에는 없음(선택 적용)

# LoRA chaining: workflow ships with a Lightning LoRA at node 89 (UNet -> 89 -> 66).
# When the caller passes lora_url, we insert a second LoraLoaderModelOnly between
# 89 and 66 so the user LoRA stacks on top of Lightning instead of replacing it.
_NODE_LIGHTNING_LORA = "89"          # existing Lightning LoraLoaderModelOnly
_NODE_MODEL_SAMPLING = "66"          # ModelSamplingAuraFlow downstream
_NODE_USER_LORA = "190"              # injected dynamically — beyond all workflow IDs (max=120)
_LORAS_DIR = "/ComfyUI/models/loras"


def _maybe_authenticated_civitai_url(url):
    """Civitai download URLs return 401 without ?token=. Auto-append CIVITAI_TOKEN
    from env (set on the RunPod endpoint template)."""
    if "civitai.com" not in url:
        return url
    if "token=" in url:
        return url
    token = os.environ.get("CIVITAI_TOKEN", "").strip()
    if not token:
        logger.warning("⚠️  Civitai LoRA URL but CIVITAI_TOKEN env var is empty — download will likely 401")
        return url
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}token={token}"


def download_lora(lora_url):
    """Download a LoRA into /ComfyUI/models/loras and return its filename.

    Idempotent: filename is derived from sha1(url) so repeat calls hit the disk
    cache. Returns just the basename — ComfyUI's LoraLoader resolves it against
    its loras dir at queue time.
    """
    fetch_url = _maybe_authenticated_civitai_url(lora_url)
    h = hashlib.sha1(lora_url.encode()).hexdigest()[:16]
    filename = f"user_lora_{h}.safetensors"
    target = os.path.join(_LORAS_DIR, filename)

    if os.path.exists(target) and os.path.getsize(target) >= 1024 * 1024:
        logger.info(f"✅ LoRA cache hit: {filename} ({os.path.getsize(target)/(1024*1024):.1f}MB)")
        return filename

    os.makedirs(_LORAS_DIR, exist_ok=True)
    # Best-effort cleanup of partial / undersized prior attempts.
    if os.path.exists(target):
        os.remove(target)

    logger.info(f"📥 Downloading user LoRA from {lora_url[:80]}…")
    t0 = time.time()
    result = subprocess.run(
        ["wget", "-O", target, "--no-verbose", "--tries=2", "--timeout=120",
         "--user-agent=kaleipix/1.0", fetch_url],
        capture_output=True, text=True, timeout=240,
    )
    if result.returncode != 0 or not os.path.exists(target):
        if os.path.exists(target):
            os.remove(target)
        raise Exception(f"LoRA download failed (rc={result.returncode}): {result.stderr[:300]}")

    sz = os.path.getsize(target)
    if sz < 1024 * 1024:
        # Civitai 401 / HF 403 typically come back as a tiny HTML error page.
        os.remove(target)
        raise Exception(f"LoRA download too small ({sz} bytes) — likely auth error or wrong URL")

    logger.info(f"✅ LoRA downloaded: {filename} ({sz/(1024*1024):.1f}MB in {time.time()-t0:.1f}s)")
    return filename


def inject_user_lora(prompt, lora_filename, lora_scale):
    """Insert a LoraLoaderModelOnly node between the Lightning LoRA (89) and the
    ModelSamplingAuraFlow (66) so the user LoRA stacks on top of Lightning."""
    if _NODE_LIGHTNING_LORA not in prompt or _NODE_MODEL_SAMPLING not in prompt:
        # Workflow shape changed upstream — bail rather than corrupt the graph.
        raise Exception(
            f"workflow missing expected LoRA chain nodes "
            f"({_NODE_LIGHTNING_LORA} or {_NODE_MODEL_SAMPLING})"
        )

    prompt[_NODE_USER_LORA] = {
        "inputs": {
            "lora_name": lora_filename,
            "strength_model": float(lora_scale),
            "model": [_NODE_LIGHTNING_LORA, 0],
        },
        "class_type": "LoraLoaderModelOnly",
        "_meta": {"title": "User LoRA"},
    }
    prompt[_NODE_MODEL_SAMPLING]["inputs"]["model"] = [_NODE_USER_LORA, 0]

# ------------------------------
# 입력 처리 유틸 (path/url/base64)
# ------------------------------
def process_input(input_data, temp_dir, output_filename, input_type):
    """입력 데이터를 처리하여 파일 경로를 반환하는 함수
    - input_type: "path" | "url" | "base64"
    """
    if input_type == "path":
        logger.info(f"📁 경로 입력 처리: {input_data}")
        return input_data
    elif input_type == "url":
        logger.info(f"🌐 URL 입력 처리: {input_data}")
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.abspath(os.path.join(temp_dir, output_filename))
        return download_file_from_url(input_data, file_path)
    elif input_type == "base64":
        logger.info("🔢 Base64 입력 처리")
        return save_base64_to_file(input_data, temp_dir, output_filename)
    else:
        raise Exception(f"지원하지 않는 입력 타입: {input_type}")

def download_file_from_url(url, output_path):
    """URL에서 파일을 다운로드하는 함수"""
    try:
        result = subprocess.run([
            'wget', '-O', output_path, '--no-verbose', url
        ], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"✅ URL에서 파일을 성공적으로 다운로드했습니다: {url} -> {output_path}")
            return output_path
        else:
            logger.error(f"❌ wget 다운로드 실패: {result.stderr}")
            raise Exception(f"URL 다운로드 실패: {result.stderr}")
    except subprocess.TimeoutExpired:
        logger.error("❌ 다운로드 시간 초과")
        raise Exception("다운로드 시간 초과")
    except Exception as e:
        logger.error(f"❌ 다운로드 중 오류 발생: {e}")
        raise Exception(f"다운로드 중 오류 발생: {e}")

def save_base64_to_file(base64_data, temp_dir, output_filename):
    """Base64 데이터를 파일로 저장하는 함수"""
    try:
        decoded_data = base64.b64decode(base64_data)
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.abspath(os.path.join(temp_dir, output_filename))
        with open(file_path, 'wb') as f:
            f.write(decoded_data)
        logger.info(f"✅ Base64 입력을 '{file_path}' 파일로 저장했습니다.")
        return file_path
    except (binascii.Error, ValueError) as e:
        logger.error(f"❌ Base64 디코딩 실패: {e}")
        raise Exception(f"Base64 디코딩 실패: {e}")

def handler(job):
    job_input = job.get("input", {})

    logger.info(f"Received job input: {job_input}")
    task_id = f"task_{uuid.uuid4()}"

    # ------------------------------
    # 이미지 입력 수집 (1개 / 2개 / 3개)
    # 지원 키: image_path | image_url | image_base64
    #         image_path_2 | image_url_2 | image_base64_2
    #         image_path_3 | image_url_3 | image_base64_3
    # ------------------------------
    image_paths = []

    for i, suffix in enumerate([ "", "_2", "_3" ], start=1):
        path_key = f"image_path{suffix}"
        url_key = f"image_url{suffix}"
        b64_key = f"image_base64{suffix}"
        fname = f"input_image_{i}.jpg"
        if path_key in job_input:
            image_paths.append(process_input(job_input[path_key], task_id, fname, "path"))
        elif url_key in job_input:
            image_paths.append(process_input(job_input[url_key], task_id, fname, "url"))
        elif b64_key in job_input:
            image_paths.append(process_input(job_input[b64_key], task_id, fname, "base64"))
        else:
            break

    num_images = len(image_paths)
    if num_images == 0:
        return {"error": "최소 1개의 이미지 입력이 필요합니다. (image_path / image_url / image_base64 중 하나)"}

    if num_images not in _WORKFLOW_FILES:
        return {"error": f"지원하는 이미지 개수는 1, 2, 3개입니다. 입력된 이미지: {num_images}개"}

    workflow_filename = _WORKFLOW_FILES[num_images]
    workflow_path = os.path.join(_WORKFLOW_BASE, workflow_filename)
    if not os.path.exists(workflow_path):
        return {"error": f"워크플로우 파일을 찾을 수 없습니다: {workflow_path}"}

    prompt = load_workflow(workflow_path)

    # 노드 번호는 각 워크플로우 JSON과 동일하게 사용
    prompt[_NODE_IMAGE_1]["inputs"]["image"] = image_paths[0]
    if num_images >= 2:
        prompt[_NODE_IMAGE_2]["inputs"]["image"] = image_paths[1]
    if num_images >= 3:
        prompt[_NODE_IMAGE_3]["inputs"]["image"] = image_paths[2]

    prompt[_NODE_PROMPT]["inputs"]["prompt"] = job_input.get("prompt", "")
    if _NODE_SEED in prompt and "seed" in job_input:
        prompt[_NODE_SEED]["inputs"]["seed"] = job_input["seed"]
    if _NODE_WIDTH in prompt and "width" in job_input:
        prompt[_NODE_WIDTH]["inputs"]["value"] = job_input["width"]
    if _NODE_HEIGHT in prompt and "height" in job_input:
        prompt[_NODE_HEIGHT]["inputs"]["value"] = job_input["height"]

    # Optional user LoRA — stacks on top of the bundled Lightning LoRA.
    lora_url = job_input.get("lora_url")
    if lora_url:
        lora_scale = job_input.get("lora_scale", 1.0)
        try:
            lora_fname = download_lora(lora_url)
            inject_user_lora(prompt, lora_fname, lora_scale)
            logger.info(f"🎨 User LoRA applied: {lora_fname} @ scale={lora_scale}")
        except Exception as e:
            # Don't fail the whole job — log and continue without LoRA so the
            # user still gets an output (with a warning) instead of an opaque error.
            logger.error(f"❌ User LoRA failed, continuing without: {e}")

    ws_url = f"ws://{server_address}:8188/ws?clientId={client_id}"
    logger.info(f"Connecting to WebSocket: {ws_url}")
    
    # 먼저 HTTP 연결이 가능한지 확인
    http_url = f"http://{server_address}:8188/"
    logger.info(f"Checking HTTP connection to: {http_url}")
    
    # HTTP 연결 확인 (최대 1분)
    max_http_attempts = 180
    for http_attempt in range(max_http_attempts):
        try:
            import urllib.request
            response = urllib.request.urlopen(http_url, timeout=5)
            logger.info(f"HTTP 연결 성공 (시도 {http_attempt+1})")
            break
        except Exception as e:
            logger.warning(f"HTTP 연결 실패 (시도 {http_attempt+1}/{max_http_attempts}): {e}")
            if http_attempt == max_http_attempts - 1:
                raise Exception("ComfyUI 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인하세요.")
            time.sleep(1)
    
    ws = websocket.WebSocket()
    # 웹소켓 연결 시도 (최대 3분)
    max_attempts = int(180/5)  # 3분 (1초에 한 번씩 시도)
    for attempt in range(max_attempts):
        try:
            ws.connect(ws_url)
            logger.info(f"웹소켓 연결 성공 (시도 {attempt+1})")
            break
        except Exception as e:
            logger.warning(f"웹소켓 연결 실패 (시도 {attempt+1}/{max_attempts}): {e}")
            if attempt == max_attempts - 1:
                raise Exception("웹소켓 연결 시간 초과 (3분)")
            time.sleep(5)
    images = get_images(ws, prompt)
    ws.close()

    # 이미지가 없는 경우 처리
    if not images:
        return {"error": "이미지를 생성할 수 없습니다."}
    
    # 첫 번째 이미지 반환
    for node_id in images:
        if images[node_id]:
            return {"image": images[node_id][0]}
    
    return {"error": "이미지를 찾을 수 없습니다."}

runpod.serverless.start({"handler": handler})