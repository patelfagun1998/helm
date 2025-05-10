import os, json, base64, asyncio, pathlib, collections, time
from typing import Literal
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import psutil  # Add this import for memory monitoring

# Suppress gRPC fork warning
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "0"

# ---  OpenAI setup (GPT‑4o & GPT‑4o‑mini) -------------
from openai import AsyncOpenAI, InternalServerError, BadRequestError
oa = AsyncOpenAI(api_key=openai_api_key)

# ---  Google GenAI setup (Gemini 2.5 Pro) -------------
import google.generativeai as genai
genai.configure(api_key=gemini_api_key)

# Use specific model version instead of preview alias
gemini = genai.GenerativeModel(
    "gemini-2.5-flash-preview-04-17",
)

Label = Literal["typically_developing", "speech_disorder", ...]  # expand to your set

def load_audio_b64(audio_path: pathlib.Path) -> str:
    """Load an MP3 file and convert it to base64."""
    return base64.b64encode(audio_path.read_bytes()).decode()

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((InternalServerError, BadRequestError)),
    reraise=True
)
async def query_gpt(audio_b64: str, model_name: str, system_prompt: str) -> Label:
    """
    Send a multimodal chat‑completion request to GPT‑4o or GPT‑4o‑mini.
    """
    try:
        resp = await oa.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Assess this recording and label it. Respond with a JSON object containing a single key 'label' with value either 'articulation' or 'phonological'. Do not include any markdown formatting or code blocks."
                        },
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": audio_b64,
                                "format": "mp3"
                            }
                        }
                    ]
                }
            ],
            max_tokens=100,
            timeout=30
        )        
        # Parse the response as JSON, handling potential markdown formatting
        content = resp.choices[0].message.content.strip()
        content = content.replace("```json", "").replace("```", "").strip()
        
        result = json.loads(content)
        if "label" not in result:
            raise ValueError(f"Response missing 'label' field: {content}")
        if result["label"] not in ["articulation", "phonological"]:
            raise ValueError(f"Invalid label value: {result['label']}")
        return result["label"]
            
    except InternalServerError as e:
        print(f"Server error from {model_name}: {str(e)}")
        print(f"Request ID: {e.response.headers.get('x-request-id', 'unknown')}")
        raise
    except BadRequestError as e:
        print(f"Bad request to {model_name}: {str(e)}")
        print(f"Request details: {e.response.json() if hasattr(e, 'response') else 'No details'}")
        raise
    except json.JSONDecodeError as e:
        print(f"Invalid JSON response from {model_name}: {content}")
        raise
    except Exception as e:
        print(f"Unexpected error in GPT {model_name}: {str(e)}")
        raise

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    reraise=True
)
async def query_gemini(audio_b64: str, system_prompt: str) -> Label:
    start_time = time.time()
    try:
        # Run the synchronous Gemini call in a thread pool
        resp = await asyncio.to_thread(
            gemini.generate_content,
            [
                system_prompt,
                {
                    "inline_data": {
                        "mime_type": "audio/mpeg",
                        "data": audio_b64,
                    }
                },
                "Assess this recording and label it."
            ],
            generation_config={
                "response_mime_type": "application/json",
            },
        )
        return json.loads(resp.text)["label"]
    except Exception as e:
        print(f"Error in Gemini: {str(e)}")
        raise

async def annotate_one(audio_path: pathlib.Path, meta_path: pathlib.Path):
    start_time = time.time()
    
    try:
        # First read the JSON to get the words
        with meta_path.open() as f:
            meta = json.load(f)
        
        # Get words and join them if it's a list
        # if disorder_type key is present, skip
        if "disorder_type" in meta:
            print(f"Skipping {audio_path.name} because disorder_type key is present")
            return
        words = meta.get('words', [])
        if isinstance(words, list):
            words = ' '.join(words)
            
        # Construct system prompt with the words
        system_prompt = (
            f"""You are a highly experienced Speech-Language Pathologist (SLP). 
An audio recording will be provided, typically consisting of a speech prompt 
from a pathologist followed by a child's repetition. 
Based on your professional expertise:

1. Assess the child's speech in the recording for signs of typical development 
or potential speech-language disorder.

2. Return a JSON object with a single key 'label' whose value is EXACTLY one of these two options:
A - 'articulation' (difficulty producing specific speech sounds correctly, such as substituting, omitting, or distorting sounds)
B - 'phonological' (difficulty understanding and using the sound system of language, affecting sounds of a particular type)

The prompt text the child is trying to repeat is as follows: {words}"""
        )

        audio_b64 = load_audio_b64(audio_path)

        # Fire off all three requests concurrently
        t4o, t4mini, tGem = await asyncio.gather(
            query_gpt(audio_b64, "gpt-4o-audio-preview", system_prompt),
            query_gpt(audio_b64, "gpt-4o-mini-audio-preview", system_prompt),
            query_gemini(audio_b64, system_prompt),
        )

        # Majority (simple mode); tie‑break in favour of GPT‑4o
        votes = [t4o, t4mini, tGem]
        majority = collections.Counter(votes).most_common(1)[0][0]

        # Update JSON with results
        meta["disorder_type"] = majority

        meta_path.write_text(json.dumps(meta, indent=2))
        # print full path
        # print(f"✓ {meta_path.resolve()}: {majority}")
    except Exception as e:
        print(f"Error processing {audio_path.name}: {str(e)}")
        meta["error"] = str(e)
        meta_path.write_text(json.dumps(meta, indent=2))
        raise

async def main(dataset_dir: str = "data"):
    dataset = pathlib.Path(dataset_dir)
    # Recursively find all mp3 files in the directory and its subdirectories
    pairs = sorted(dataset.rglob("*.mp3"))
    tasks = [
        annotate_one(mp3, mp3.with_suffix(".json"))
        for mp3 in pairs
    ]
    
    # Optimal batch size for M2 MacBook Air
    # Considering:
    # - 8GB/16GB unified memory
    # - 8 CPU cores
    # - 3 API calls per file
    # - Audio data size
    batch_size = 8  # Process 8 files at a time
    
    for i in range(0, len(tasks), batch_size):
        print(f"\nProcessing batch {i//batch_size + 1} of {(len(tasks) + batch_size - 1)//batch_size}")
        try:
            await asyncio.gather(*tasks[i : i + batch_size])
        except Exception as e:
            print(f"Error in batch {i//batch_size + 1}: {str(e)}")
            continue

if __name__ == "__main__":
    directory = '/Users/fagunpatel/Library/CloudStorage/GoogleDrive-fagunpatel1998@gmail.com/My Drive/SpeechData/scenarios/speech_disorder/processed-core-uxssd'
    asyncio.run(main(directory))

