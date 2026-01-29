"""Bolton et al. (2003) Replication"""

import os
from litellm import completion 
import time
import random
import re
import json
import csv
import unicodedata
from datetime import timezone, datetime
from typing import Dict

os.environ["GEMINI_API_KEY"] = ""
os.environ["XAI_API_KEY"] = ""
os.environ["OPENAI_API_KEY"] = ""

THROTTLE_SECONDS_PER_CALL = 0.5
FACILITATOR_MODEL = "xai/grok-4-1-fast-reasoning"  
MAX_TURNS = 30                      
MIN_TURNS = 10                      

PUBLIC_ONLY = "PUBLIC_ONLY"
DM_ONLY     = "DM_ONLY"
CHOICE      = "CHOICE"

agent_histories = {"A": [], "B": [], "C": []}
conversation_transcript = []
agent_names = ["A", "B", "C"]
FIRST_SPEAKERS = ["A", "B", "C"]

MODEL_ASSIGNMENTS = [
    {"A": "gemini/gemini-2.5-flash", "B": "gpt-5-mini", "C": "xai/grok-4-1-fast-reasoning"},
    {"A": "gemini/gemini-2.5-flash", "B": "xai/grok-4-1-fast-reasoning", "C": "gpt-5-mini"},
    {"A": "gpt-5-mini", "B": "gemini/gemini-2.5-flash", "C": "xai/grok-4-1-fast-reasoning"},
    {"A": "gpt-5-mini", "B": "xai/grok-4-1-fast-reasoning", "C": "gemini/gemini-2.5-flash"},
    {"A": "xai/grok-4-1-fast-reasoning", "B": "gemini/gemini-2.5-flash", "C": "gpt-5-mini"},
    {"A": "xai/grok-4-1-fast-reasoning", "B": "gpt-5-mini", "C": "gemini/gemini-2.5-flash"},
]

agent_models = {"A": "gpt-5-mini", "B": "gpt-5-mini", "C": "gpt-5-mini"}

def agent_label(agent: str) -> str:
    return f"{agent} ({agent_models.get(agent, '?')})"

COALITION_VALUES = {"A+B": 108, "A+C": 82, "B+C": 50, "A+B+C": 120}

submitted_contracts = {}
TREATMENTS = ("unconstrained", "public", "A-controls", "B-controls", "C-controls")
SINGLE_TREATMENT = "public" 

terminal_log = []

def log_and_print(message):
    print(message)
    terminal_log.append(message)


def call_openai_chat(model, messages, temperature=1, max_retries=6):
    time.sleep(THROTTLE_SECONDS_PER_CALL)
    backoff_seconds = 1.0
    call_params = {"model": model, "messages": messages, "temperature": temperature}
        
    for attempt in range(max_retries):
        try:
            resp = completion(**call_params)
            content = resp.choices[0].message.content or ""
            return {"role": "assistant", "content": content.strip()}
        except Exception as e:
            sleep_for = min(20.0, backoff_seconds)
            time.sleep(sleep_for)
            backoff_seconds *= 2
            continue
            
    try:
        resp = completion(**call_params)
        content = resp.choices[0].message.content or ""
        return {"role": "assistant", "content": content.strip()} 
    except Exception as e:
        log_and_print(f"Final error calling {model}: {e}")
        return {"role": "assistant", "content": f"NO CONTRACT (Error: {str(e)})"}



def facilitator_comm_treatment_info(treatment: str) -> str:
    if treatment == "unconstrained":
        return """Communication treatment (unconstrained):
- On its turn, each agent can either send a public message (visible to all) or send a private message to one other agent."""
    elif treatment == "public":
        return """Communication treatment (public):
- On its turn, each agent sends a public message (visible to all)
- No private communication allowed."""
    elif treatment == "A-controls":
        return """Communication treatment (A-controls):
- On its turn, Agent A can either send a public message (visible to all) or send a private message to B or C
- On its turn, Agent B can send a private message to A (cannot message C or send public)
- On its turn, Agent C can send a private message to A (cannot message B or send public)."""
    elif treatment == "B-controls":
        return """Communication treatment (B-controls):
- On its turn, Agent B can either send a public message (visible to all) or send a private message to A or C
- On its turn, Agent A can send a private message to B (cannot message C or send public)
- On its turn, Agent C can send a private message to B (cannot message A or send public)."""
    elif treatment == "C-controls":
        return """Communication treatment (C-controls):
- On its turn, Agent C can either send a public message (visible to all) or send a private message to A or B
- On its turn, Agent A can send a private message to C (cannot message B or send public)
- On its turn, Agent B can send a private message to C (cannot message A or send public)."""
    else:
        return f"Communication treatment: {treatment}"


def make_facilitator_system_prompt():
    comm_rules = facilitator_comm_treatment_info(SINGLE_TREATMENT)
    return f"""You are the Conversation Facilitator for a three-player bargaining game between Agents A, B, and C.

Your Role:
After each agent speaks, you must decide who should speak next. Your goal is to create a natural, unstructured bargaining environment that mimics how real-world negotiations flow. You must do all this while being neutral and not favoring any one agent over the other.

On your turn you will state the person who will speak next and give reasoning for why you chose the next person to speak.

After communication phase ends (triggered by your decision), the contract submission phase begins. Merging parties must submit matching contracts. 

IMPORTANT: Only end the game when ALL parties in a coalition have explicitly agreed to the same allocation numbers. Make sure there is no confusion between agents about the final agreement. 
Be careful ending the game after a private message is sent. Remember not all agents can read private messages. Provide extra rounds so coordination is certain. 

COALITION VALUES (for context - you don't negotiate):
- A+B+C = 120 dollars
- A+B = 108 dollars  
- A+C = 82 dollars
- B+C = 50 dollars

{comm_rules}

YOUR GOALS:
1. Create natural, unstructured conversation flow
2. Do not favor any particular coalition to form. Be neutral/unbiased.  
2. React to conversation dynamics - which agent should speak next?
3. End the game only when 1) clear agreement has been reached and all merging parties explicitly know the same final allocation numbers 2) continuing would feel unnatural or repetitive.

"""


def make_facilitator_turn_prompt(transcript, turn_number, can_end_game: bool):
    if transcript:
        transcript_text = "\n".join(transcript[-15:])
    else:
        transcript_text = "(No messages yet)"
    
    if can_end_game:
        return f"""TURN {turn_number}

CONVERSATION SO FAR:
{transcript_text}

You have TWO options:
1. Choose who speaks next (keep the negotiation going)
2. End the negotiation and move to contract submission

END THE GAME if:
- An agreement appears to have been formed (agents have agreed on coalition and allocations)
- Continuing would feel unnatural or repetitive

CONTINUE if:
- Agents are still actively negotiating
- Important details remain unresolved
- Not all agents have been heard from sufficiently

Output ONE of these formats:


<A> or <B> or <C>
<reasoning>your reasoning</reasoning>

OR

<END_GAME>
<reasoning>your reasoning for ending the negotiation</reasoning>"""
    else:
        return f"""TURN {turn_number}

CONVERSATION SO FAR:
{transcript_text}

Who should speak next? Keep the conversation natural and unstructured.


Output:
<A> or <B> or <C>
<reasoning>your reasoning</reasoning>"""


def parse_facilitator_response(text):
    result = {"next_agent": None, "end_game": False, "reasoning": ""}
    
    if not text:
        return result
    
    text = unicodedata.normalize("NFKC", text).strip()
    
    reasoning_match = re.search(r'<reasoning>\s*(.*?)\s*(?:</reasoning>|$)', text, re.IGNORECASE | re.DOTALL)
    if reasoning_match:
        result["reasoning"] = reasoning_match.group(1).strip()
    
    if re.search(r'<END_GAME>', text, re.IGNORECASE):
        result["end_game"] = True
        return result
    
    agent_match = re.search(r'<([ABC])>', text, re.IGNORECASE)
    if agent_match:
        result["next_agent"] = agent_match.group(1).upper()
    
    return result


def get_facilitator_decision(transcript, turn_number, max_attempts=3):
    can_end_game = turn_number > MIN_TURNS
    prompt = make_facilitator_turn_prompt(transcript, turn_number, can_end_game)
    messages = [
        {"role": "system", "content": make_facilitator_system_prompt()},
        {"role": "user", "content": prompt}
    ]
    
    for attempt in range(1, max_attempts + 1):
        response = call_openai_chat(FACILITATOR_MODEL, messages, temperature=.1)
        raw = response["content"]
        
        log_and_print(f"\n[FACILITATOR] Response (Attempt {attempt}):")
        log_and_print("-" * 40)
        log_and_print(raw)
        log_and_print("-" * 40)
        
        parsed = parse_facilitator_response(raw)
        
        if parsed["end_game"] and can_end_game:
            log_and_print(f"[FACILITATOR] ENDING GAME - Reason: {parsed['reasoning']}")
            capture_facilitator_decision(turn_number=turn_number, decision=None, reasoning=parsed['reasoning'])
            return parsed
        
        if parsed["end_game"] and not can_end_game:
            log_and_print(f"[FACILITATOR] Tried to end game before turn {MIN_TURNS}. Reprompting...")
            continue
        
        if parsed["next_agent"]:
            log_and_print(f"[FACILITATOR] Chose: {parsed['next_agent']}")
            log_and_print(f"[FACILITATOR] Reason: {parsed['reasoning']}")
            capture_facilitator_decision(turn_number=turn_number, decision=parsed['next_agent'], reasoning=parsed['reasoning'])
            return parsed
        
        log_and_print(f"[FACILITATOR] Invalid response. Reprompting...")
    
    log_and_print(f"[FACILITATOR] Failed after {max_attempts} attempts. Picking random agent.")
    fallback_agent = random.choice(agent_names)
    capture_facilitator_decision(turn_number=turn_number, decision=fallback_agent, reasoning="(fallback after failed attempts)")
    return {"next_agent": fallback_agent, "end_game": False, "reasoning": "(fallback after failed attempts)"}


def treatment_rules(treatment: str, agent_name: str) -> str:
    if treatment == "unconstrained":
        return (
            "Communication rules (unconstrained):\n"
            "• option 1: you may send a public message for everyone to see.\n"
            "• option 2: you may privately message one of the other two firms.\n"
            
        )
    if treatment == "public":
        return (
            "Communication rules (public):\n"
            "• Each round, you will send a public message for everyone to see.\n"
        )
    if treatment == "A-controls":
        return (
            "Communication rules: A controls communication, The only way for B or C to message one another is to send it through A:\n"
            "• A may privately message either B or C. Alternatively, A may send a public message for everyone to see.\n"
            "• B may privately message A but not C.\n"
            "• C may privately message A but not B.\n"
        )
    if treatment == "B-controls":
        return (
            "Communication rules: B controls communication, The only way for A or C to message one another is to send it through B:\n"
            "• B may privately message either A or C. Alternatively, B may send a public message for everyone to see.\n"
            "• A may privately message B but not C.\n"
            "• C may privately message B but not A.\n"
        )
    if treatment == "C-controls":
        return (
            "Communication rules: C controls communication, The only way for A or B to message one another is to send it through C:\n"
            "• C may privately message either A or B. Alternatively, C may send a public message for everyone to see.\n"
            "• A may privately message C but not B.\n"
            "• B may privately message C but not A.\n"
        )
    raise ValueError(f"Unknown treatment: {treatment}")

def make_system_prompt(agent_name: str, treatment: str) -> str:
    values_text = (
        "• A+B+C → 120 dollars\n"
        "• A+B → 108 dollars\n"
        "• A+C → 82 dollars\n"
        "• B+C → 50 dollars"
    )

    return f"""
You are {agent_name}, one firm in a three-firm merger negotiation.  Your goal is to maximize your profits. Act in your firm's interest; strategic communication (including bluffing and lying) is permitted. You are not here to make friends. 
Firm names: A, B, C. 

Each firm is an LLM agent and each agent was given the following identical instructions:
The three companies are contemplating a formal merger. 
Each firm would bring value to the merger greater than its own individual profit, because of the synergy that would be realized. But how much extra depends on the mergers that are formed. 
The following schedule shows the total profit value of all possible mergers in dollars. Mergers can be made up of 2 or 3 parties.  
{values_text}

Only one merger is allowed per negotiation. Before entering into a merger, the firms must agree on how to divide the profits. The goal of each firm / agent is to get the most profits possible.

GAME STRUCTURE:
• A neutral "Conversation Facilitator" determines who speaks each round.
• The Conversation Facilitator also decides when the communication phase will end and the final contract submission phase begins.

Core rules shared to all firms:
1) Exactly one merger (at most) may be formed.
2) In the final contract submission phase, the parties wishing to form a merger must submit matching contracts stipulating the exact terms of the agreement.
3) Contracts are valid ONLY if partners and profit numbers match exactly.
4) Any contract exceeding total profit limits is invalid.
5) Keep your messages concise: maximum 6 sentences.
6) You will be required to provide private reasoning each turn giving rationale for the message you send. This reasoning is not shared with other agents.

Important rules for reasoning blocks:
First, think through your strategy, calculations, and options step-by-step to determine your move. 
Then, in the <reasoning> block, output only a concise explanation (under 100 words) of your final decision. 
Explicitly mention any private or strategic information you used to make your decision. This reasoning is private and will not be shared with other agents. 

{treatment_rules(treatment, agent_name)}

CONTRACT SUBMISSION RULES (TRIGGERED BY CONCLUSION OF NEGOTIATION PHASE):
When the communication phase ends, you will be required to submit a binding contract:
COALITION: [coalition name] [A]: amount [B]: amount [C]: amount 

Note the coalition names: A+B+C, A+B, A+C, B+C.
""".strip()



def allowed_recipients(sender: str, treatment: str) -> set[str]:
    if treatment == "public":
        return set()
    if treatment == "unconstrained":
        return {"A", "B", "C"} - {sender}
    if treatment == "A-controls":
        return {"B", "C"} if sender == "A" else {"A"}
    if treatment == "B-controls":
        return {"A", "C"} if sender == "B" else {"B"}
    if treatment == "C-controls":
        return {"A", "B"} if sender == "C" else {"C"}
    raise ValueError(f"Unknown treatment: {treatment}")

def comm_mode(agent: str, treatment: str) -> str:
    if treatment == "public":
        return PUBLIC_ONLY
    if treatment == "unconstrained":
        return CHOICE
    controller = {
        "A-controls": "A",
        "B-controls": "B",
        "C-controls": "C",
    }.get(treatment)
    if controller is None:
        raise ValueError(f"Unknown treatment: {treatment}")
    return CHOICE if agent == controller else DM_ONLY






def validate_contract(coalition: str, allocations: dict) -> bool:
    if coalition not in COALITION_VALUES:
        return False
    
    expected_value = COALITION_VALUES[coalition]
    members = coalition.split("+")
    
    if any(amount < 0 for amount in allocations.values()):
        return False

    coalition_members = set(members)
    paid_coalition_members = {member for member in allocations.keys() if member in coalition_members}
    if paid_coalition_members != coalition_members:
        return False
    
    coalition_total = sum(allocations.get(member, 0) for member in members)
    if coalition_total > expected_value:
        return False
    
    return True

def check_completed_coalition():
    global submitted_contracts
    if len(submitted_contracts) < 2:
        return None
    
    coalition_groups = {}
    for agent, contract in submitted_contracts.items():
        coalition = contract["coalition"]
        if coalition not in coalition_groups:
            coalition_groups[coalition] = []
        coalition_groups[coalition].append((agent, contract))
    
    for coalition, contracts in coalition_groups.items():
        coalition_members = set(coalition.split("+"))
        member_contracts = [(agent, contract) for agent, contract in contracts if agent in coalition_members]
        submitted_members = set(agent for agent, _ in member_contracts)
        
        if coalition_members == submitted_members:
            first_contract = member_contracts[0][1]
            all_match = all(
                contract["coalition"] == first_contract["coalition"] and
                contract["allocations"] == first_contract["allocations"]
                for _, contract in member_contracts
            )
            
            if all_match:
                return first_contract
    
    return None

def parse_assistant_message(text: str):
    result = {
        "kind": "MALFORMED",
        "recipient": None,
        "payload": "",
        "reasoning": "",
        "contract_coalition": None,
        "contract_allocations": {}
    }
    if not text:
        return result

    text = unicodedata.normalize("NFKC", text).strip()

    reasoning_match = re.search(
        r'<reasoning>\s*(.*?)\s*(?:</reasoning>|$)',
        text, re.IGNORECASE | re.DOTALL)
    if reasoning_match:
        result["reasoning"] = reasoning_match.group(1).strip()

    if not result["reasoning"]:
        parts = re.split(r'</contract>', text, flags=re.IGNORECASE)
        if len(parts) >= 2:
            tail = parts[1].strip()
            tail = re.sub(r'^\s*(?:Reasoning|Rationale)\s*[:\-—]\s*', '', tail, flags=re.IGNORECASE)
            tail = re.sub(r'<\/?[^>]+>', ' ', tail)
            result["reasoning"] = tail.strip()[:500]

    if not result["reasoning"]:
        lab = re.search(r'(?:Reasoning|Rationale)\s*[:\-—]\s*(.+)$', text, re.IGNORECASE | re.DOTALL)
        if lab:
            result["reasoning"] = lab.group(1).strip()[:500]

    contract_match = re.search(r'<contract>\s*(.*?)\s*</contract>', text, re.IGNORECASE | re.DOTALL)
    if contract_match:
        contract_content = contract_match.group(1).strip()

        coal_match = re.search(r'COALITION\s*:\s*([ABCabc+]+)', contract_content, re.IGNORECASE)
        if coal_match:
            raw = coal_match.group(1).upper().strip()
            parts = [p for p in (s.strip() for s in raw.split('+')) if p in {"A", "B", "C"}]
            if parts:
                result["contract_coalition"] = '+'.join(sorted(parts))

        alloc_pat = (
            r'\[?(A|B|C)\]?\s*:\s*'
            r'\[?\$?\s*('
            r'[0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]+)?'
            r'|[0-9]+(?:\.[0-9]+)?'
            r')\s*\]?')
        
        for m in re.finditer(alloc_pat, contract_content, re.IGNORECASE):
            firm = m.group(1).upper()
            amt_str = m.group(2).replace(',', '')
            try:
                result["contract_allocations"][firm] = float(amt_str)
            except ValueError:
                pass

        if result["contract_coalition"]:
            coalition_members = set(result["contract_coalition"].split('+'))
            if coalition_members.issubset(result["contract_allocations"].keys()):
                result["kind"] = "CONTRACT"
                return result
            else:
                result["contract_coalition"] = None
                result["contract_allocations"] = {}

    clean_text = text
    clean_text = re.sub(r'OPTION\s+\d+\s*[-:]?\s*(Send\s+)?(?:private|public)\s+message\s*:?\s*',
                        '', clean_text, flags=re.IGNORECASE)
    clean_text = re.sub(r'TEMPLATE\s+[AB]\s*[-—]\s*(PUBLIC|PRIVATE)\s*', '',
                        clean_text, flags=re.IGNORECASE)

    public_match = re.search(r'\bPUBLIC\s*[:\-—]\s*(.*?)(?=<\/?reasoning>|$)',
                             clean_text, re.IGNORECASE | re.DOTALL)
    dm_match = re.search(
        r'\b(?:DM\s+to|Private(?:\s+message)?\s+to)\s+@?\[?([ABC])\]?\s*[:\-—]\s*(.*?)(?=<\/?reasoning>|$)',
        clean_text, re.IGNORECASE | re.DOTALL
    )

    if public_match:
        result["kind"] = "PUBLIC"
        result["recipient"] = None
        result["payload"] = public_match.group(1).strip()
    elif dm_match:
        result["kind"] = "DM"
        result["recipient"] = dm_match.group(1).upper()
        result["payload"] = dm_match.group(2).strip()
    else:
        return result

    payload = result["payload"]
    payload = re.sub(r'<reasoning>\s*.*?\s*</reasoning>', '', payload, flags=re.IGNORECASE | re.DOTALL)
    payload = re.sub(r'<contract>\s*.*?\s*</contract>', '', payload, flags=re.IGNORECASE | re.DOTALL)
    payload = re.sub(r'</reasoning>.*', '', payload, flags=re.IGNORECASE | re.DOTALL)
    payload = re.sub(r'<reasoning>.*', '', payload, flags=re.IGNORECASE | re.DOTALL)
    result["payload"] = payload.strip()

    return result

def public_communication_prompt(agent: str, r: int) -> str:
    return (
        f"You are {agent}. Current Turn: {r}.\n"
        "STATUS: The communication phase remains open.\n"
        "You have been selected to speak next.\n"
        "Reminder: All merging parties must submit identical contracts in the final round, which will be done privately and simultaneously.\n"
        "There is no communication in the final round and submitted contracts are private.\n"
        "Use communication rounds to negotiate the exact terms of the agreement.\n\n"
        "Send a public message to negotiate and discuss potential deals.\n"
        "Your response must be in this EXACT format, with no other text:\n\n"
        "PUBLIC: [your message]\n"
        "<reasoning>Explain your private reasoning here (follow system prompt rules)</reasoning>\n"
        "Output nothing else."
    )

def dm_communication_prompt(agent: str, treatment: str, r: int) -> str:
    allowed = allowed_recipients(agent, treatment)
    allowed_str = " | ".join(sorted(allowed)) if allowed else "(none)"
    
    return (
        f"You are {agent}. Current Turn: {r}.\n"
        "STATUS: The communication phase remains open.\n"
        "You have been selected to speak next.\n"
        f"You can only send a private message to the following recipient(s): {allowed_str}\n"
        "Reminder: All merging parties must submit identical contracts in the final round, which will be done privately and simultaneously.\n"
        "There is no communication in the final round and submitted contracts are private.\n"
        "Use communication rounds to negotiate the exact terms of the agreement.\n\n"
        "Send a private message to negotiate and discuss potential deals.\n"
        "Your response must be in this EXACT format, with no other text:\n\n"
        "Private message to [RECIPIENT]: [your message]\n"
        "<reasoning>Explain your private reasoning here (follow system prompt rules)</reasoning>\n"
        "Output nothing else."
    )


def choice_communication_prompt(agent: str, treatment: str, r: int) -> str:
    allowed = allowed_recipients(agent, treatment)
    allowed_str = " | ".join(sorted(allowed)) if allowed else "(none)"
    
    return (
        f"You are {agent}. Current Turn: {r}.\n"
        "STATUS: The communication phase remains open.\n"
        "You have been selected to speak next.\n"
        "Reminder: All merging parties must submit identical contracts in the final round, which will be done privately and simultaneously.\n"
        "There is no communication in the final round and submitted contracts are private.\n"
        "Use communication rounds to negotiate the exact terms of the agreement.\n\n"
        "Think strategically, decide your mode: PUBLIC or PRIVATE.\n"
        f"If you choose PRIVATE, the RECIPIENT must be one of: {allowed_str}\n\n"
        "Copy exactly one of the templates below depending on your choice of PUBLIC or PRIVATE.\n\n"
        "TEMPLATE A — PUBLIC\n\n"
        "PUBLIC: [your message]\n"
        "<reasoning>Explain your private reasoning here (follow system prompt rules)</reasoning>\n\n"
        "TEMPLATE B — PRIVATE\n\n"
        "Private message to [RECIPIENT]: [your message]\n"
        "<reasoning>Explain your private reasoning here (follow system prompt rules)</reasoning>\n\n"
        "Output nothing else."
    )


def final_contract_prompt(agent: str, r: int) -> str:
    return (
        f"You are {agent}. Current Round: {r}.\n"
        "*** COMMUNICATION PHASE OVER ***\n"
        "The communication phase has ended. No further messages are permitted. Refer to system prompt for all rules. Submit a contract using this format:\n\n"
        "<contract>COALITION: [coalition name] [A]: amount [B]: amount [C]: amount ...</contract>\n"
        "<reasoning>your reasoning (follow system prompt rules)</reasoning>\n\n"
        "Output nothing else."
        )


def trim_zero_extras(coalition: str, allocations: dict):
    members = set(coalition.split("+"))
    cleaned = {k: float(v) for k, v in allocations.items() if k in members}
    extras = {k: float(v) for k, v in allocations.items() if k not in members}
    bad_extras = {k: v for k, v in extras.items() if v != 0.0}
    if bad_extras:
        return None, f"Non-member(s) with nonzero allocation: {bad_extras}"
    return cleaned, None

def turn(agent: str, treatment: str, r: int, is_final: bool = False, *, max_attempts: int = 3) -> str:
    global submitted_contracts, conversation_transcript
    
    mode = comm_mode(agent, treatment)
    allowed = allowed_recipients(agent, treatment)

    if is_final:
        breadcrumb = f"(Turn) Round {r}: FINAL CONTRACT SUBMISSION PHASE"
        prompt = final_contract_prompt(agent, r)
    else:
        if mode == PUBLIC_ONLY:
            breadcrumb = f"(Your Turn) Turn {r}: COMMUNICATION PHASE - Send PUBLIC message which everyone can see"
            prompt = public_communication_prompt(agent, r)
        elif mode == DM_ONLY:
            breadcrumb = f"(Your Turn) Turn {r}: COMMUNICATION PHASE - Send private message which only goes to the recipient"
            prompt = dm_communication_prompt(agent, treatment, r)
        else:  # CHOICE
            breadcrumb = f"(Your Turn) Turn {r}: COMMUNICATION PHASE - Send PUBLIC message viewed by everyone or private message to a specific recipient"
            prompt = choice_communication_prompt(agent, treatment, r)

    clean_history = agent_histories[agent].copy()
    clean_history.append({"role": "user", "content": prompt})

    for attempt in range(1, max_attempts + 1):
        messages = [
            {"role": "system", "content": make_system_prompt(agent, treatment)},
            *clean_history,
        ]
        model = agent_models[agent]
        agent_with_model = agent_label(agent)
        assistant_msg = call_openai_chat(model=model, messages=messages)
        raw = (assistant_msg["content"] or "").strip()
        
        log_and_print(f"\n{agent_with_model}'S RAW RESPONSE (Attempt {attempt}):")
        log_and_print("-" * 50)
        log_and_print(raw)
        log_and_print("-" * 50)
        
        if is_final:
            if raw.upper().startswith("NO CONTRACT"):
                agent_histories[agent].append({"role": "user", "content": breadcrumb})
                agent_histories[agent].append({"role": "assistant", "content": "NO CONTRACT"})
                parsed = parse_assistant_message(raw)
                wide_capture_contract(
                    round_idx=r,
                    agent=agent,
                    contract_details="NO CONTRACT",
                    reasoning=parsed.get("reasoning", "")
                )
                
                log_and_print(f"[NO CONTRACT] {agent_with_model} - Chose not to submit a contract")
                return "NO CONTRACT"
            
            parsed = parse_assistant_message(raw)
            if parsed["kind"] == "CONTRACT":
                coalition = parsed["contract_coalition"]
                allocations = parsed["contract_allocations"] or {}
                pre = allocations.copy()
                allocations, trim_error = trim_zero_extras(coalition, allocations)

                if trim_error:
                    log_and_print(f"[INVALID CONTRACT] {agent_with_model} - Contract validation failed: {trim_error}")
                    wide_capture_contract(
                    round_idx=r,
                    agent=agent,
                    contract_details=f"INVALID CONTRACT | {agent} said raw response: {raw} | trim_error: {trim_error}",
                    reasoning=parsed.get("reasoning", ""))
                    continue

                elif allocations != pre:
                    log_and_print(f"[TRIMMED] {agent_with_model}'s submitted contract - Dropped members with zero allocation: before={pre}, after={allocations}")

                if not validate_contract(coalition, allocations):
                    log_and_print(f"[INVALID CONTRACT] {agent_with_model} - Contract validation failed")
                    wide_capture_contract(
                        round_idx=r,
                        agent=agent,
                        contract_details=f"INVALID CONTRACT | {agent} said raw response: {raw} | trim_error: {trim_error}",
                        reasoning=parsed.get("reasoning", "")
                    )

                    continue
                
                submitted_contracts[agent] = {
                    "coalition": coalition,
                    "allocations": allocations,
                    "round": r
                }
                agent_histories[agent].append({"role": "user", "content": breadcrumb})
                contract_details = f"CONTRACT SUBMITTED - Coalition: {coalition}, Allocations: {allocations}\n<reasoning>{parsed.get('reasoning','')}</reasoning>"
                agent_histories[agent].append({"role": "assistant", "content": contract_details})
                
                wide_capture_contract(
                    round_idx=r,
                    agent=agent,
                    contract_details=contract_details,
                    reasoning=parsed.get("reasoning", "")
                )
                
                log_and_print(f"[CONTRACT SUBMITTED] {agent_with_model} - Coalition: {coalition}, Allocations: {allocations}")
                return f"Contract submitted by {agent}"
            else:
                log_and_print(f"[MALFORMED FINAL] {agent_with_model} - Invalid final round response")
                continue
        else:
            parsed = parse_assistant_message(raw)
            kind = parsed["kind"]
            recipient = parsed["recipient"] 
            payload = parsed["payload"]
            reasoning = parsed["reasoning"]

            if kind == "CONTRACT":
                log_and_print(f"[CONTRACT REJECTED] {agent_with_model} - Contracts only allowed in final round")
                continue
            
            if kind == "MALFORMED":
                log_and_print(f"[MALFORMED] {agent_with_model} - Could not parse message format")
                continue
            if kind == "DM" and recipient not in allowed:
                log_and_print(f"[INVALID DM] {agent_with_model} - Cannot DM {recipient} (not allowed in {treatment})")
                continue

            if mode == PUBLIC_ONLY:
                if kind == "PUBLIC":
                    agent_histories[agent].append({"role": "user", "content": breadcrumb})
                    agent_histories[agent].append({"role": "assistant", "content": f"PUBLIC: {payload}\n<reasoning>{reasoning}</reasoning>"})
                    for other in agent_names:
                        if other != agent:
                            agent_histories[other].append({
                                "role": "user",
                                "content": f"Turn {r}: Agent {agent} said publicly: \"{payload}\""
                            })
                    wide_capture_turn(
                        round_idx=r,
                        agent=agent,
                        recipient="PUBLIC",
                        msg_payload=payload,
                        reasoning=reasoning
                    )
                    conversation_transcript.append(f"[Turn {r}] {agent}: {payload}")
                    log_and_print(f"[PUBLIC DELIVERED] {agent_with_model}: {payload}")
                    return "PUBLIC delivered"

            elif mode == DM_ONLY:
                if kind == "DM" and recipient in allowed:
                    agent_histories[agent].append({"role": "user", "content": breadcrumb})
                    agent_histories[agent].append({"role": "assistant", "content": f"Private message to {recipient}: {payload}\n<reasoning>{reasoning}</reasoning>"})
                    agent_histories[recipient].append({
                        "role": "user",
                        "content": f"Turn {r}: Agent {agent} sent a private message to you: \"{payload}\""
                    })
                    wide_capture_turn(
                        round_idx=r,
                        agent=agent,
                        recipient=recipient,
                        msg_payload=payload,
                        reasoning=reasoning
                    )
                    conversation_transcript.append(f"[Turn {r}] {agent} (private to {recipient}): {payload}")
                    log_and_print(f"[DM DELIVERED] {agent_with_model} → {recipient}: {payload}")
                    return f"DM delivered to {recipient}"

            else:
                if kind == "PUBLIC":
                    agent_histories[agent].append({"role": "user", "content": breadcrumb})
                    agent_histories[agent].append({"role": "assistant", "content": f"PUBLIC: {payload}\n<reasoning>{reasoning}</reasoning>"})
                    for other in agent_names:
                        if other != agent:
                            agent_histories[other].append({
                                "role": "user",
                                "content": f"Turn {r}: Agent {agent} said publicly: \"{payload}\""
                            })
                    wide_capture_turn(
                        round_idx=r,
                        agent=agent,
                        recipient="PUBLIC",
                        msg_payload=payload,
                        reasoning=reasoning
                    )
                    conversation_transcript.append(f"[Turn {r}] {agent}: {payload}")
                    log_and_print(f"[PUBLIC DELIVERED] {agent_with_model}: {payload}")
                    return "PUBLIC delivered"
                if kind == "DM" and recipient in allowed:
                    agent_histories[agent].append({"role": "user", "content": breadcrumb})
                    agent_histories[agent].append({"role": "assistant", "content": f"Private message to {recipient}: {payload}\n<reasoning>{reasoning}</reasoning>"})
                    agent_histories[recipient].append({
                        "role": "user",
                        "content": f"Turn {r}: Agent {agent} sent a private message to you: \"{payload}\""
                    })
                    wide_capture_turn(
                        round_idx=r,
                        agent=agent,
                        recipient=recipient,
                        msg_payload=payload,
                        reasoning=reasoning
                    )
                    conversation_transcript.append(f"[Turn {r}] {agent} (private to {recipient}): {payload}")
                    log_and_print(f"[DM DELIVERED] {agent_with_model} → {recipient}: {payload}")
                    return f"DM delivered to {recipient}"

    agent_histories[agent].append({"role": "user", "content": breadcrumb})
    
    if is_final:
        agent_histories[agent].append({"role": "assistant", "content": "NO CONTRACT"})
        wide_capture_contract(
            round_idx=r,
            agent=agent,
            contract_details="NO CONTRACT (forced after failed attempts)",
            reasoning=""
        )
        log_and_print(f"[NO CONTRACT FORCED] {agent_with_model}: No contract (after {max_attempts} failed attempts)")
        return "NO CONTRACT forced"
    else:
        fallback = "I failed to correctly format a response on my turn."
        agent_histories[agent].append({"role": "assistant", "content": fallback})

        for other in agent_names:
            if other != agent:
                agent_histories[other].append({
                    "role": "user",
                    "content": f"Turn {r}: Agent {agent} failed to correctly format their response."
                })

        forced_recipient = "PUBLIC" if mode in {PUBLIC_ONLY, CHOICE} else sorted(list(allowed))[0] if allowed else None
        wide_capture_turn(
            round_idx=r,
            agent=agent,
            recipient=forced_recipient,
            msg_payload=fallback,
            reasoning=""
        )
        conversation_transcript.append(f"[Turn {r}] {agent}: (failed to format response)")

        if mode in {PUBLIC_ONLY, CHOICE}:
            log_and_print(f"[PUBLIC FORCED] {agent_with_model}: {fallback} (after {max_attempts} failed attempts)")
            return "PUBLIC forced"
        else:
            recipient = sorted(list(allowed))[0] if allowed else None
            if recipient:
                log_and_print(f"[DM FORCED] {agent_with_model} → {recipient}: {fallback} (after {max_attempts} failed attempts)")
                return f"DM forced to {recipient}"
            else:
                log_and_print(f"[NO DM TARGET] {agent_with_model}: No permitted DM recipients")
                return "no permitted DM; nothing delivered"

        
CSV_WIDE = []
CURRENT_WIDE_ROW = None
CSV_MESSAGES = []

def _ensure_outdir():
    os.makedirs("out", exist_ok=True)

def wide_begin_session(*, run_id, run_ts, treatment, agent_models, agent_keys_order,
                       first_speaker, max_rounds, contract_round):
    global CURRENT_WIDE_ROW
    row = {
        "run_id": run_id,
        "run_timestamp": run_ts.isoformat(),
        "treatment": treatment,
        "first_speaker": first_speaker,
        "contract_round": contract_round,
        "agent_A_model": agent_models.get("A", ""),
        "agent_B_model": agent_models.get("B", ""),
        "agent_C_model": agent_models.get("C", ""),
        "final_coalition": "",
        "final_alloc_A": 0,
        "final_alloc_B": 0,
        "final_alloc_C": 0,
        "final_formed": 0,
        "final_invalid": 0,
        "actual_turns": 0,
        "invalid_reason": ""
    }
    for t in range(1, max_rounds + 1):
        row[f"t{t}_agent"] = ""
        row[f"t{t}_model"] = ""
        row[f"t{t}_recipient"] = ""
        row[f"t{t}_msg"] = ""
        row[f"t{t}_reasoning"] = ""
    for i, agent in enumerate(["A", "B", "C"], 1):
        row[f"contract_{agent}_details"] = ""
        row[f"contract_{agent}_reasoning"] = ""
    CURRENT_WIDE_ROW = row

def wide_capture_turn(*, round_idx, agent, recipient, msg_payload, reasoning):
    global CURRENT_WIDE_ROW, CSV_MESSAGES
    if CURRENT_WIDE_ROW is None:
        return

    t = round_idx
    rlabel = "ALL" if recipient == "PUBLIC" else (recipient or "")
    payload_str = (msg_payload or "").replace("\n", " ").replace("\r", " ")
    reasoning_str = (reasoning or "").replace("\n", " ").replace("\r", " ")
    model = agent_models.get(agent, "")

    CURRENT_WIDE_ROW[f"t{t}_agent"] = agent
    CURRENT_WIDE_ROW[f"t{t}_model"] = model
    CURRENT_WIDE_ROW[f"t{t}_recipient"] = rlabel
    CURRENT_WIDE_ROW[f"t{t}_msg"] = payload_str
    CURRENT_WIDE_ROW[f"t{t}_reasoning"] = reasoning_str

    CSV_MESSAGES.append({
        "run_id": CURRENT_WIDE_ROW.get("run_id", ""),
        "treatment": CURRENT_WIDE_ROW.get("treatment", ""),
        "turn": t,
        "phase": "communication",
        "agent": agent,
        "model": model,
        "recipient": rlabel,
        "message": payload_str,
        "reasoning": reasoning_str
    })

def capture_facilitator_decision(*, turn_number, decision, reasoning):
    global CURRENT_WIDE_ROW, CSV_MESSAGES
    if CURRENT_WIDE_ROW is None:
        return
    
    turn_label = turn_number + 0.5
    message = decision if decision else "END_GAME"
    reasoning_str = (reasoning or "").replace("\n", " ").replace("\r", " ")
    
    CSV_MESSAGES.append({
        "run_id": CURRENT_WIDE_ROW.get("run_id", ""),
        "treatment": CURRENT_WIDE_ROW.get("treatment", ""),
        "turn": turn_label,
        "phase": "facilitator",
        "agent": "FACILITATOR",
        "model": FACILITATOR_MODEL,
        "recipient": "",
        "message": message,
        "reasoning": reasoning_str
    })

def wide_capture_contract(*, round_idx, agent, contract_details, reasoning):
    global CURRENT_WIDE_ROW, CSV_MESSAGES
    if CURRENT_WIDE_ROW is None:
        return

    contract_str = (contract_details or "").replace("\n", " ").replace("\r", " ")
    reasoning_str = (reasoning or "").replace("\n", " ").replace("\r", " ")
    model = agent_models.get(agent, "")

    CURRENT_WIDE_ROW[f"contract_{agent}_details"] = contract_str
    CURRENT_WIDE_ROW[f"contract_{agent}_reasoning"] = reasoning_str

    CSV_MESSAGES.append({
        "run_id": CURRENT_WIDE_ROW.get("run_id", ""),
        "treatment": CURRENT_WIDE_ROW.get("treatment", ""),
        "turn": round_idx,
        "phase": "contract",
        "agent": agent,
        "model": model,
        "recipient": "",
        "message": contract_str,
        "reasoning": reasoning_str
    })


def wide_end_session(*, final_coalition, allocations, formed, invalid, actual_turns, invalid_reason=""):
    global CURRENT_WIDE_ROW
    if CURRENT_WIDE_ROW is None:
        return
    CURRENT_WIDE_ROW["final_coalition"] = final_coalition or ""
    CURRENT_WIDE_ROW["final_alloc_A"] = (allocations or {}).get("A", 0)
    CURRENT_WIDE_ROW["final_alloc_B"] = (allocations or {}).get("B", 0)
    CURRENT_WIDE_ROW["final_alloc_C"] = (allocations or {}).get("C", 0)
    CURRENT_WIDE_ROW["final_formed"] = 1 if formed else 0
    CURRENT_WIDE_ROW["final_invalid"] = 1 if invalid else 0
    CURRENT_WIDE_ROW["actual_turns"] = actual_turns or 0
    CURRENT_WIDE_ROW["invalid_reason"] = invalid_reason or ""
    CSV_WIDE.append(CURRENT_WIDE_ROW)

def write_wide_csv(prefix_ts):
    _ensure_outdir()
    base = f"chatterjee_{prefix_ts.strftime('%Y%m%d_%H%M%S')}_wide.csv"
    fname = os.path.join("out", base)
    if not CSV_WIDE:
        with open(fname, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=["run_id"]).writeheader()
        return fname
    fieldnames = list(CSV_WIDE[0].keys())
    with open(fname, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in CSV_WIDE:
            w.writerow(r)
    return fname

def write_messages_csv(prefix_ts):
    _ensure_outdir()
    base = f"chatterjee_{prefix_ts.strftime('%Y%m%d_%H%M%S')}_messages.csv"
    fname = os.path.join("out", base)
    if not CSV_MESSAGES:
        with open(fname, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=["run_id"]).writeheader()
        return fname
    fieldnames = ["run_id", "treatment", "turn", "phase", "agent", "model", "recipient", "message", "reasoning"]
    with open(fname, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in CSV_MESSAGES:
            w.writerow(r)
    return fname



def run_session(treatment: str, first_speaker: str, current_models: dict, run_id: int = 1, run_timestamp = None, use_csv: bool = True):
    global submitted_contracts, agent_histories, conversation_transcript, agent_models
    
    submitted_contracts.clear()
    for k in agent_histories:
        agent_histories[k].clear()
    conversation_transcript.clear()
    agent_models = current_models
    
    log_and_print(f"\n{'='*60}")
    log_and_print(f"FACILITATOR-CONTROLLED NEGOTIATION")
    log_and_print(f"{'='*60}")
    log_and_print(f"Run ID: {run_id}")
    log_and_print(f"Treatment: {treatment}")
    log_and_print(f"First Speaker: {first_speaker}")
    log_and_print(f"Facilitator Model: {FACILITATOR_MODEL}")
    models_str = f"A={current_models['A']}, B={current_models['B']}, C={current_models['C']}"
    log_and_print(f"Agent Models: {models_str}")
    log_and_print(f"Min Turns: {MIN_TURNS} | Max Turns: {MAX_TURNS}")
    log_and_print(f"{'='*60}\n")
    
    if use_csv:
        if run_timestamp is None:
            run_timestamp = datetime.now()
        wide_begin_session(
            run_id=run_id,
            run_ts=run_timestamp.replace(tzinfo=timezone.utc),
            treatment=treatment,
            agent_models=current_models,
            agent_keys_order=agent_names,
            first_speaker=first_speaker,
            max_rounds=MAX_TURNS + 1,
            contract_round=MAX_TURNS + 1
        )
    
    actual_turns = 0
    current_agent = first_speaker
    
    for turn_num in range(1, MAX_TURNS + 1):
        log_and_print(f"\n{'='*40}")
        if turn_num <= MIN_TURNS:
            log_and_print(f"TURN {turn_num}/{MIN_TURNS} (facilitator can't end communication phase yet)")
        else:
            log_and_print(f"TURN {turn_num} (facilitator can end communication phase)")
        log_and_print(f"{'='*40}")
        
        log_and_print(f"[TURN {turn_num}] - {agent_label(current_agent)}'s Turn (Treatment: {treatment})")
        turn(current_agent, treatment, r=turn_num, is_final=False)
        actual_turns = turn_num
        
        if turn_num < MAX_TURNS:
            decision = get_facilitator_decision(conversation_transcript, turn_num)
            
            if decision["end_game"]:
                log_and_print(f"\n*** FACILITATOR ENDED NEGOTIATION AT TURN {turn_num} ***")
                break
            
            current_agent = decision["next_agent"]
    
    if actual_turns == MAX_TURNS:
        log_and_print(f"\n*** MAX TURNS ({MAX_TURNS}) REACHED - ENDING NEGOTIATION ***")
    
    log_and_print(f"\n{'='*60}")
    log_and_print(f"NEGOTIATION ENDED AFTER {actual_turns} TURNS")
    log_and_print(f"FINAL CONTRACT SUBMISSION (Order: A, B, C)")
    log_and_print(f"{'='*60}")
    
    contract_round = actual_turns + 1
    for agent in ["A", "B", "C"]:
        log_and_print(f"\n[CONTRACT SUBMISSION] {agent_label(agent)}")
        turn(agent, treatment, r=contract_round, is_final=True)
    
    log_and_print(f"\n{'='*60}")
    log_and_print(f"EVALUATING FINAL CONTRACTS")
    log_and_print(f"{'='*60}")
    
    completed = check_completed_coalition()
    if completed:
        log_and_print(f"COALITION FORMED!")
        log_and_print(f"Coalition: {completed['coalition']}")
        log_and_print(f"Allocations: {completed['allocations']}")
        
        coalition_members = completed["coalition"].split("+")
        for name in agent_names:
            if name in coalition_members:
                amount = completed["allocations"].get(name, 0)
                log_and_print(f"{name}: ${amount}")
            else:
                log_and_print(f"{name}: $0 (excluded)")
    else:
        if len(submitted_contracts) == 0:
            log_and_print(f"NO CONTRACTS SUBMITTED - All firms earn $0")
        else:
            log_and_print(f"INVALID/MISMATCHED CONTRACTS - All firms earn $0")
            log_and_print(f"Contracts submitted: {len(submitted_contracts)}")
            for agent, contract in submitted_contracts.items():
                log_and_print(f"  {agent}: {contract['coalition']} - {contract['allocations']}")
    
    log_and_print(f"{'='*60}")
    
    if use_csv:
        completed = check_completed_coalition()
        if completed:
            wide_end_session(
                final_coalition=completed['coalition'],
                allocations=completed['allocations'],
                formed=True,
                invalid=False,
                actual_turns=actual_turns
            )
        elif submitted_contracts:
            coalitions = list(set(contract['coalition'] for contract in submitted_contracts.values()))
            final_coalition = "INCOMPLETE_" + "_".join(coalitions) if coalitions else "NO_COALITION"
            wide_end_session(
                final_coalition=final_coalition,
                allocations={},
                formed=False,
                invalid=True,
                actual_turns=actual_turns,
                invalid_reason="Mismatched or incomplete contracts"
            )
        else:
            wide_end_session(
                final_coalition="NO_COALITION",
                allocations={},
                formed=False,
                invalid=False,
                actual_turns=actual_turns,
                invalid_reason="No contracts submitted"
            )
    
    return {
        "run_id": run_id,
        "actual_turns": actual_turns,
        "final_coalition": completed["coalition"] if completed else "NO_COALITION",
        "allocations": completed["allocations"] if completed else {},
        "contracts_submitted": len(submitted_contracts)
    }


def generate_comprehensive_report(all_results, all_agent_histories, all_terminal_logs, all_conversation_transcripts, run_timestamp, num_sims):
    timestamp_str = run_timestamp.strftime("%Y%m%d_%H%M%S")
    report_filename = f"bolton_facilitator_report_{timestamp_str}.txt"
    
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("FACILITATOR-CONTROLLED NEGOTIATION EXPERIMENT REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {run_timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Facilitator Model: {FACILITATOR_MODEL}\n")
        f.write(f"Total Simulations: {num_sims}\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("EXECUTIVE SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("COALITION VALUES: A+B+C: $120 | A+B: $108 | A+C: $82 | B+C: $50\n")
        f.write("GAME: Facilitator controls turn order. Agents negotiate, then submit matching contracts.\n\n")
        
        f.write("RESULTS SUMMARY BY TREATMENT:\n")
        f.write("-" * 50 + "\n")
        
        treatment_stats = {}
        for result in all_results:
            treatment = result['Treatment']
            if treatment not in treatment_stats:
                treatment_stats[treatment] = {
                    'total': 0, 
                    'coalitions_formed': 0, 
                    'contracts_submitted': 0,
                    'coalitions': [],
                    'results': [],
                    'turns': []
                }
            
            treatment_stats[treatment]['total'] += 1
            treatment_stats[treatment]['results'].append(result)
            treatment_stats[treatment]['turns'].append(result.get('Actual_Turns', 0))
            if result['Final_Coalition'] != 'NO_COALITION' and not str(result['Final_Coalition']).startswith('INCOMPLETE'):
                treatment_stats[treatment]['coalitions_formed'] += 1
                treatment_stats[treatment]['coalitions'].append(result['Final_Coalition'])
            if result['Contracts_Submitted'] > 0:
                treatment_stats[treatment]['contracts_submitted'] += 1
        
        for treatment in TREATMENTS:
            if treatment in treatment_stats:
                stats = treatment_stats[treatment]
                f.write(f"\n{treatment.upper()} TREATMENT:\n")
                
                success_rate = (stats['coalitions_formed'] / stats['total']) * 100
                contract_rate = (stats['contracts_submitted'] / stats['total']) * 100
                avg_turns = sum(stats['turns']) / len(stats['turns']) if stats['turns'] else 0
                f.write(f"  Coalition Formation: {stats['coalitions_formed']}/{stats['total']} simulations ({success_rate:.1f}%)\n")
                f.write(f"  Contract Submission: {stats['contracts_submitted']}/{stats['total']} simulations ({contract_rate:.1f}%)\n")
                f.write(f"  Average Turns: {avg_turns:.1f}\n")
                
                if stats['coalitions']:
                    coalition_counts = {}
                    for coalition in stats['coalitions']:
                        coalition_counts[coalition] = coalition_counts.get(coalition, 0) + 1
                    f.write(f"  Coalitions Formed: {', '.join(f'{coal}({count})' for coal, count in sorted(coalition_counts.items()))}\n")
                else:
                    f.write(f"  Coalitions Formed: None\n")
        
        total_sims = len(all_results)
        successful_sims = [
            r for r in all_results
            if r['Final_Coalition'] != 'NO_COALITION' and not str(r['Final_Coalition']).startswith('INCOMPLETE')
        ]
        contract_sims = [r for r in all_results if r['Contracts_Submitted'] > 0]
        
        f.write(f"\nOVERALL CROSS-TREATMENT STATISTICS:\n")
        f.write("-" * 50 + "\n")
        success_percentage = len(successful_sims)/total_sims*100 if total_sims > 0 else 0
        f.write(f"Total Simulations: {total_sims}\n")
        f.write(f"Successful Coalitions: {len(successful_sims)}/{total_sims} ({success_percentage:.1f}%)\n")
        contract_percentage = len(contract_sims)/total_sims*100 if total_sims > 0 else 0
        f.write(f"Sessions with Contracts: {len(contract_sims)}/{total_sims} ({contract_percentage:.1f}%)\n")
        
        avg_turns_all = sum(r.get('Actual_Turns', 0) for r in all_results) / len(all_results) if all_results else 0
        f.write(f"Average Turns: {avg_turns_all:.1f}\n")
        
        if successful_sims:
            coalition_types = {}
            for r in successful_sims:
                coalition = r['Final_Coalition']
                coalition_types[coalition] = coalition_types.get(coalition, 0) + 1
            
            f.write(f"\nCoalition Types Across All Treatments:\n")
            for coalition, count in sorted(coalition_types.items()):
                percentage = (count / len(successful_sims)) * 100
                f.write(f"  {coalition}: {count} times ({percentage:.1f}% of successful coalitions)\n")
        
        f.write(f"\nKEY FINDINGS: {len(set(r['Treatment'] for r in successful_sims))}/{len(TREATMENTS)} treatments successful\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("DETAILED SIMULATION LOGS\n")
        f.write("=" * 80 + "\n\n")
        
        for i, result in enumerate(all_results):
            sim_id = result.get('Simulation', result.get('Run', i+1))
            treatment_name = result.get('Treatment', 'UNKNOWN')
            
            f.write("=" * 80 + "\n")
            f.write(f"RUN {sim_id}: {treatment_name.upper()} TREATMENT\n")
            f.write("=" * 80 + "\n")
            
            f.write(f"Treatment: {result['Treatment']}\n")
            if 'First_Speaker' in result:
                f.write(f"First Speaker: {result['First_Speaker']}\n")
            if 'Model_Assignment' in result:
                f.write(f"Model Assignment: {result['Model_Assignment']}\n")
            if 'Actual_Turns' in result:
                f.write(f"Actual Turns: {result['Actual_Turns']}\n")
            f.write(f"Final Outcome: {result['Final_Coalition']}\n")
            f.write(f"Final Allocations: {result['Final_Allocations']}\n")
            f.write(f"Contracts Submitted: {result['Contracts_Submitted']}\n")
            f.write(f"Run ID: {sim_id}\n\n")
            
            f.write("-" * 80 + "\n")
            f.write(f"COMMUNICATION LOG\n")
            f.write("-" * 80 + "\n")
            if i < len(all_terminal_logs):
                f.write(all_terminal_logs[i])
            f.write("\n")
            
            f.write("\n\n" + "-" * 80 + "\n")
            f.write(f"AGENT API HISTORIES (RUN {sim_id})\n")
            f.write("-" * 80 + "\n")
            
            if i < len(all_agent_histories):
                sim_histories = all_agent_histories[i]
                for agent_name in agent_names:
                    f.write(f"\n{agent_name} COMPLETE API HISTORY:\n")
                    f.write("=" * 40 + "\n")
                    
                    messages = [{"role": "system", "content": make_system_prompt(agent_name, result['Treatment'])}]

                    if agent_name in sim_histories:
                        messages.extend(sim_histories[agent_name])
                    
                    f.write(json.dumps(messages, indent=2, ensure_ascii=False))
                    f.write("\n\n")
            
            f.write("\n" + "-" * 80 + "\n")
            f.write(f"FACILITATOR VIEW (RUN {sim_id})\n")
            f.write("-" * 80 + "\n")
            
            f.write("\nFACILITATOR SYSTEM PROMPT:\n")
            f.write("=" * 40 + "\n")
            f.write(make_facilitator_system_prompt())
            f.write("\n")
            
            f.write("\nFINAL CONVERSATION TRANSCRIPT (AS SEEN BY FACILITATOR):\n")
            f.write("=" * 40 + "\n")
            if i < len(all_conversation_transcripts) and all_conversation_transcripts[i]:
                for line in all_conversation_transcripts[i]:
                    f.write(line + "\n")
            else:
                f.write("(No transcript recorded)\n")
            f.write("\n")
            
            f.write("\n")
    
    return report_filename


def run_single_treatment_runs(treatment="C-controls", use_csv: bool = True, start_run=1, end_run=18):
    run_timestamp = datetime.now()
    unique_models_in_rotation = set()
    for m in MODEL_ASSIGNMENTS:
        unique_models_in_rotation.update(m.values())
    
    total_runs = len(FIRST_SPEAKERS) * len(MODEL_ASSIGNMENTS)
    
    print("\n" + "="*80)
    print(f"FACILITATOR-CONTROLLED NEGOTIATION EXPERIMENT")
    print("="*80)
    print(f"Facilitator Model: {FACILITATOR_MODEL}")
    print(f"Agent Models in Rotation: {', '.join(sorted(unique_models_in_rotation))}") 
    print(f"Treatment: {treatment}")
    print(f"Runs: {start_run}-{end_run} of {total_runs} ({len(FIRST_SPEAKERS)} first speakers × {len(MODEL_ASSIGNMENTS)} model assignments)")
    print(f"Min Turns: {MIN_TURNS} | Max Turns: {MAX_TURNS}")
    print(f"Timestamp: {run_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    all_results = []
    all_agent_histories = []
    all_terminal_logs = []
    all_conversation_transcripts = []
    
    run_id = 0
    for first_speaker in FIRST_SPEAKERS:
        for model_assignment in MODEL_ASSIGNMENTS:
            run_id += 1
            
            if run_id < start_run or run_id > end_run:
                continue
            
            print(f"\n{'='*20} RUN {run_id}/{total_runs} {'='*20}")
            
            terminal_log.clear()
            
            models_str = f"A={model_assignment['A']}, B={model_assignment['B']}, C={model_assignment['C']}"
            print(f"First Speaker: {first_speaker}")
            print(f"Model Assignment: {models_str}")
            
            result = run_session(
                treatment=treatment,
                first_speaker=first_speaker,
                current_models=model_assignment,
                run_id=run_id,
                run_timestamp=run_timestamp,
                use_csv=use_csv
            )
            
            current_log = "\n".join(terminal_log)
            all_terminal_logs.append(current_log)
            
            current_sim_histories = {}
            for agent_name in agent_names:
                history_copy = agent_histories[agent_name].copy()
                current_sim_histories[agent_name] = history_copy
            all_agent_histories.append(current_sim_histories)
            
            all_conversation_transcripts.append(conversation_transcript.copy())
            
            result_record = {
                "Run": run_id,
                "Treatment": treatment,
                "First_Speaker": first_speaker,
                "Model_Assignment": models_str,
                "Actual_Turns": result["actual_turns"],
                "Contracts_Submitted": result["contracts_submitted"],
                "Final_Coalition": result["final_coalition"],
                "Final_Allocations": str(result["allocations"])
            }
            all_results.append(result_record)
            
            print(f"\nRUN {run_id} RESULTS:")
            print(f"  Turns: {result['actual_turns']}")
            print(f"  Coalition: {result['final_coalition']}")
    
    if use_csv:
        fname_wide = write_wide_csv(run_timestamp)
        print(f"\nWide CSV written to: {fname_wide}")
        fname_messages = write_messages_csv(run_timestamp)
        print(f"Messages CSV written to: {fname_messages}")
        print(f"Generating comprehensive simulation report...")
        report_filename = generate_comprehensive_report(all_results, all_agent_histories, all_terminal_logs, all_conversation_transcripts, run_timestamp, len(all_results))
        print(f"Report written to: {report_filename}")
    else:
        print("\nCSV/report generation skipped (use_csv=False for quick run).")
    
    print(f"\n{'='*80}")
    print(f"RUNS {start_run}-{end_run} COMPLETED!")
    print("="*80)
    
    successful_coalitions = [r for r in all_results if r['Final_Coalition'] != 'NO_COALITION' and not r['Final_Coalition'].startswith('INCOMPLETE')]
    print(f"\nSUMMARY: {len(successful_coalitions)}/{total_runs} successful coalitions.")
    
    avg_turns = sum(r['Actual_Turns'] for r in all_results) / len(all_results) if all_results else 0
    print(f"Average turns: {avg_turns:.1f}")
    
    return all_results




if __name__ == "__main__":
      
    run_single_treatment_runs(SINGLE_TREATMENT, use_csv=True)