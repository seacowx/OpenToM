B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

def convert_to_llama_prompt(chatgpt_prompt: list) -> str:
    llama_prompt = ""
    llama_prompt += f"<s>{B_INST} {B_SYS}\n{chatgpt_prompt[0]['content']}\n{E_SYS}\n"
    print(chatgpt_prompt)
    for idx, content_dict in enumerate(chatgpt_prompt):

        # skip system prompt
        if idx == 0:
            continue

        # user instruction
        elif idx == 1:
            llama_prompt += f"{content_dict['content']} {E_INST}"

        # assistant response
        else:
            if content_dict['role'] == 'user':
                llama_prompt += f"<s> {B_INST}{content_dict['content']} {E_INST}"
            elif content_dict['role'] == 'assistant':
                llama_prompt += f"{content_dict['content']} </s>"

    return llama_prompt
