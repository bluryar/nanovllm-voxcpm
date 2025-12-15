import requests
import base64
import json
import time
import numpy as np
import wave
import argparse
import concurrent.futures
import threading
import statistics

import os

# Global statistics
stats = {
    "total_requests": 0,
    "success_requests": 0,
    "failed_requests": 0,
    "latencies": [],
    "ttfts": [],  # 添加TTFT统计
    "rtfs": [],   # 添加RTF统计
    "total_bytes_received": 0,
    "errors": []
}
stats_lock = threading.Lock()

def test_single_request(request_id, url, payload, output_dir="output_audio", sample_rate=16000):
    start_time = time.time()
    try:
        # print(f"[Req {request_id}] Sending request...")
        response = requests.post(url, data=payload, stream=True, verify=False, timeout=300) # Long timeout for TTS
        response.raise_for_status()
        
        first_chunk_time = None
        full_audio_data = b""
        
        # Iterate over lines for SSE
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                if decoded_line.startswith("data: "):
                    if first_chunk_time is None:
                        first_chunk_time = time.time()
                    
                    json_str = decoded_line[6:]  # Remove "data: " prefix
                    if json_str.strip() == "[DONE]":
                        break
                        
                    try:
                        data = json.loads(json_str)
                        
                        # Handle different event types
                        if data.get("type") == "speech.audio.delta":
                            audio_chunk_b64 = data.get("audio")
                            if audio_chunk_b64:
                                audio_chunk = base64.b64decode(audio_chunk_b64)
                                full_audio_data += audio_chunk
                        
                        elif data.get("type") == "speech.audio.done":
                            break
                            
                    except json.JSONDecodeError:
                        continue
        
        end_time = time.time()
        latency = end_time - start_time
        # 确保TTFT计算正确
        ttft = (first_chunk_time - start_time) if first_chunk_time else latency
        
        # Calculate RTF (Real Time Factor)
        # Audio duration in seconds = total_samples / sample_rate
        # total_samples = total_bytes / (channels * sample_width)
        # Assuming 1 channel, 16-bit (2 bytes)
        audio_duration = len(full_audio_data) / (2 * 1 * sample_rate)
        rtf = latency / audio_duration if audio_duration > 0 else 0
        
        with stats_lock:
            stats["total_requests"] += 1
            stats["success_requests"] += 1
            stats["latencies"].append(latency)
            stats["ttfts"].append(ttft)  # 添加TTFT记录
            stats["rtfs"].append(rtf)    # 添加RTF记录
            stats["total_bytes_received"] += len(full_audio_data)
            
        print(f"[Req {request_id}] Success. TTFT: {ttft:.2f}s, Total: {latency:.2f}s, Audio: {audio_duration:.2f}s, RTF: {rtf:.2f}")
        
        # Save audio for every request
        os.makedirs(output_dir, exist_ok=True)
        save_audio(full_audio_data, f"{output_dir}/output_{request_id}.wav", sample_rate=sample_rate)

    except Exception as e:
        with stats_lock:
            stats["total_requests"] += 1
            stats["failed_requests"] += 1
            stats["errors"].append(str(e))
        print(f"[Req {request_id}] Failed: {e}")

def save_audio(audio_data, filename, sample_rate=16000):
    if len(audio_data) > 0:
        try:
            # audio_data is already int16 bytes from SSE
            # We just need to save it directly to WAV
            
            # If the data came from float32 SSE (old bug), we would need conversion.
            # But since we fixed app.py to send int16, audio_data here is raw int16 bytes.
            
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2) # 16-bit
                wf.setframerate(sample_rate)
                wf.writeframes(audio_data)
            print(f"Saved {filename}")
        except Exception as e:
            print(f"Error saving audio {filename}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Concurrent TTS Testing Script")
    parser.add_argument("--url", type=str, default="http://localhost:8000/v1/audio/speech", help="API URL")
    parser.add_argument("--concurrency", type=int, default=1, help="Number of concurrent requests")
    parser.add_argument("--requests", type=int, default=1, help="Total number of requests to send")
    parser.add_argument("--sample-rate", type=int, default=44100, help="Sample rate for saving audio (default: 44100)")
    parser.add_argument("--text", type=str, default="你好，我是dabin，这是流式语音合成的并发测试。是的，这段代码完全支持流式推理。让我为你分析一下代码中的流式实现机制：", help="Text to synthesize")
    
    args = parser.parse_args()
    
    payload = {
        "input": args.text,
        "voice": "dabin",
        "response_format": "mp3", 
        "speed": 1.0,
        "stream_format": "sse"
    }
    
    if args.requests < args.concurrency:
        args.requests = args.concurrency
        print(f"Adjusting total requests to match concurrency: {args.requests}")

    print(f"Starting test with {args.concurrency} threads, {args.requests} total requests.")
    print(f"Target URL: {args.url}")
    print(f"Sample Rate: {args.sample_rate} Hz")
    
    start_total = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [executor.submit(test_single_request, i, args.url, payload, sample_rate=args.sample_rate) for i in range(args.requests)]
        concurrent.futures.wait(futures)
        
    end_total = time.time()
    duration = end_total - start_total
    
    print("\n" + "="*40)
    print("Test Summary")
    print("="*40)
    print(f"Total Duration: {duration:.2f}s")
    print(f"Total Requests: {stats['total_requests']}")
    print(f"Successful: {stats['success_requests']}")
    print(f"Failed: {stats['failed_requests']}")
    
    if stats['latencies']:
        avg_latency = statistics.mean(stats['latencies'])
        p95_latency = statistics.quantiles(stats['latencies'], n=20)[18] if len(stats['latencies']) >= 20 else max(stats['latencies'])
        print(f"Average Latency: {avg_latency:.2f}s")
        print(f"P95 Latency: {p95_latency:.2f}s")
    
    # 添加TTFT统计部分
    if stats['ttfts']:
        avg_ttft = statistics.mean(stats['ttfts'])
        p95_ttft = statistics.quantiles(stats['ttfts'], n=20)[18] if len(stats['ttfts']) >= 20 else max(stats['ttfts'])
        min_ttft = min(stats['ttfts'])
        max_ttft = max(stats['ttfts'])
        
        print("\nTTFT Statistics:")
        print(f"  Average: {avg_ttft:.3f}s")
        print(f"  P95: {p95_ttft:.3f}s")
        print(f"  Min: {min_ttft:.3f}s")
        print(f"  Max: {max_ttft:.3f}s")

    # 添加RTF统计部分
    if stats['rtfs']:
        avg_rtf = statistics.mean(stats['rtfs'])
        p95_rtf = statistics.quantiles(stats['rtfs'], n=20)[18] if len(stats['rtfs']) >= 20 else max(stats['rtfs'])
        min_rtf = min(stats['rtfs'])
        max_rtf = max(stats['rtfs'])
        
        print("\nRTF Statistics:")
        print(f"  Average: {avg_rtf:.3f}")
        print(f"  P95: {p95_rtf:.3f}")
        print(f"  Min: {min_rtf:.3f}")
        print(f"  Max: {max_rtf:.3f}")
    
    print(f"\nThroughput: {stats['success_requests'] / duration:.2f} req/s")
    print(f"Total Bytes Received: {stats['total_bytes_received'] / 1024:.2f} KB")
    
    if stats['errors']:
        print("\nErrors encountered:")
        for err in set(stats['errors']):
            print(f"- {err}")

if __name__ == "__main__":
    requests.packages.urllib3.disable_warnings()
    main()