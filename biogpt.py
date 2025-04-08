from transformers import AutoTokenizer, AutoModelForCausalLM

# Load tokenizer and model
model_name = "microsoft/biogpt-large-pubmedqa"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def ask_medical_question(question, max_new_tokens=500):
    prompt = f"Question: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.95
    )
    
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer[len(prompt):].strip()


question = "Can antibiotics treat viral infections?"
answer = ask_medical_question(question)
print("Q:", question)
print("A:", answer)


if __name__ == "__main__":
    while True:
        q = input("\nAsk a medical question (or type 'exit'): ")
        if q.lower() == 'exit':
            break
        print("Answer:", ask_medical_question(q))
