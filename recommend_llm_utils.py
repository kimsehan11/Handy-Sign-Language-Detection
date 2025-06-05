from transformers import pipeline

def load_llm_pipeline():
    # 최초 실행 시 모델 다운로드가 필요할 수 있음
    pipe = pipeline("text-generation", model="agentlans/pythia-14m-sentences")
    return pipe

def get_llm_sentence_recommendation(prefix, pipe, max_length=50):
    # prefix를 이어서 자연스러운 문장 생성
    result = pipe(prefix, max_length=max_length, num_return_sequences=1, do_sample=False)
    generated = result[0]['generated_text']
    # prefix 이후의 tail만 추출
    tail = generated[len(prefix):]
    return tail
