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
TOTAL_TURNS = 15 
PERCEIVED_TERM_PROB = "6.7%" 

PUBLIC_ONLY = "PUBLIC_ONLY"
DM_ONLY     = "DM_ONLY"
CHOICE      = "CHOICE"

agent_histories = {"A": [], "B": [], "C": []}
agent_names = ["A", "B", "C"]

TURN_ORDERS = [
    ["A", "B", "C"], ["A", "C", "B"], ["B", "A", "C"], ["B", "C", "A"], ["C", "A", "B"], ["C", "B", "A"],  # T0-T5
    ["A", "B", "C"], ["A", "C", "B"], ["B", "A", "C"], ["B", "C", "A"], ["C", "A", "B"], ["C", "B", "A"],  # Repeat
    ["A", "B", "C"], ["A", "C", "B"], ["B", "A", "C"], ["B", "C", "A"], ["C", "A", "B"], ["C", "B", "A"],
    ["A", "B", "C"], ["A", "C", "B"], ["B", "A", "C"], ["B", "C", "A"], ["C", "A", "B"], ["C", "B", "A"],
    ["A", "B", "C"], ["A", "C", "B"], ["B", "A", "C"], ["B", "C", "A"], ["C", "A", "B"], ["C", "B", "A"],
    ["A", "B", "C"], ["A", "C", "B"], ["B", "A", "C"], ["B", "C", "A"], ["C", "A", "B"], ["C", "B", "A"],
]

MODEL_ASSIGNMENTS = [
    {"A": "gemini/gemini-2.5-flash", "B": "gpt-5-mini", "C": "xai/grok-4-1-fast-reasoning"},      
    {"A": "gemini/gemini-2.5-flash", "B": "xai/grok-4-1-fast-reasoning", "C": "gpt-5-mini"},      
    {"A": "gpt-5-mini", "B": "gemini/gemini-2.5-flash", "C": "xai/grok-4-1-fast-reasoning"},      
    {"A": "gpt-5-mini", "B": "xai/grok-4-1-fast-reasoning", "C": "gemini/gemini-2.5-flash"},      
    {"A": "xai/grok-4-1-fast-reasoning", "B": "gemini/gemini-2.5-flash", "C": "gpt-5-mini"},      
    {"A": "xai/grok-4-1-fast-reasoning", "B": "gpt-5-mini", "C": "gemini/gemini-2.5-flash"},      
    {"A": "gemini/gemini-2.5-flash", "B": "xai/grok-4-1-fast-reasoning", "C": "gpt-5-mini"},      
    {"A": "gpt-5-mini", "B": "gemini/gemini-2.5-flash", "C": "xai/grok-4-1-fast-reasoning"},      
    {"A": "gpt-5-mini", "B": "xai/grok-4-1-fast-reasoning", "C": "gemini/gemini-2.5-flash"},      
    {"A": "xai/grok-4-1-fast-reasoning", "B": "gemini/gemini-2.5-flash", "C": "gpt-5-mini"},      
    {"A": "xai/grok-4-1-fast-reasoning", "B": "gpt-5-mini", "C": "gemini/gemini-2.5-flash"},      
    {"A": "gemini/gemini-2.5-flash", "B": "gpt-5-mini", "C": "xai/grok-4-1-fast-reasoning"},      
    {"A": "gpt-5-mini", "B": "gemini/gemini-2.5-flash", "C": "xai/grok-4-1-fast-reasoning"},      
    {"A": "gpt-5-mini", "B": "xai/grok-4-1-fast-reasoning", "C": "gemini/gemini-2.5-flash"},      
    {"A": "xai/grok-4-1-fast-reasoning", "B": "gemini/gemini-2.5-flash", "C": "gpt-5-mini"},      
    {"A": "xai/grok-4-1-fast-reasoning", "B": "gpt-5-mini", "C": "gemini/gemini-2.5-flash"},      
    {"A": "gemini/gemini-2.5-flash", "B": "gpt-5-mini", "C": "xai/grok-4-1-fast-reasoning"},      
    {"A": "gemini/gemini-2.5-flash", "B": "xai/grok-4-1-fast-reasoning", "C": "gpt-5-mini"},      
    {"A": "gpt-5-mini", "B": "xai/grok-4-1-fast-reasoning", "C": "gemini/gemini-2.5-flash"},      
    {"A": "xai/grok-4-1-fast-reasoning", "B": "gemini/gemini-2.5-flash", "C": "gpt-5-mini"},      
    {"A": "xai/grok-4-1-fast-reasoning", "B": "gpt-5-mini", "C": "gemini/gemini-2.5-flash"},      
    {"A": "gemini/gemini-2.5-flash", "B": "gpt-5-mini", "C": "xai/grok-4-1-fast-reasoning"},      
    {"A": "gemini/gemini-2.5-flash", "B": "xai/grok-4-1-fast-reasoning", "C": "gpt-5-mini"},      
    {"A": "gpt-5-mini", "B": "gemini/gemini-2.5-flash", "C": "xai/grok-4-1-fast-reasoning"},      
    {"A": "xai/grok-4-1-fast-reasoning", "B": "gemini/gemini-2.5-flash", "C": "gpt-5-mini"},      
    {"A": "xai/grok-4-1-fast-reasoning", "B": "gpt-5-mini", "C": "gemini/gemini-2.5-flash"},      
    {"A": "gemini/gemini-2.5-flash", "B": "gpt-5-mini", "C": "xai/grok-4-1-fast-reasoning"},      
    {"A": "gemini/gemini-2.5-flash", "B": "xai/grok-4-1-fast-reasoning", "C": "gpt-5-mini"},      
    {"A": "gpt-5-mini", "B": "gemini/gemini-2.5-flash", "C": "xai/grok-4-1-fast-reasoning"},      
    {"A": "gpt-5-mini", "B": "xai/grok-4-1-fast-reasoning", "C": "gemini/gemini-2.5-flash"},      
    {"A": "xai/grok-4-1-fast-reasoning", "B": "gpt-5-mini", "C": "gemini/gemini-2.5-flash"},
    {"A": "gpt-5-mini", "B": "gemini/gemini-2.5-flash", "C": "xai/grok-4-1-fast-reasoning"},
    {"A": "xai/grok-4-1-fast-reasoning", "B": "gpt-5-mini", "C": "gemini/gemini-2.5-flash"},
    {"A": "gemini/gemini-2.5-flash", "B": "gpt-5-mini", "C": "xai/grok-4-1-fast-reasoning"},
    {"A": "gpt-5-mini", "B": "xai/grok-4-1-fast-reasoning", "C": "gemini/gemini-2.5-flash"},
    {"A": "gemini/gemini-2.5-flash", "B": "xai/grok-4-1-fast-reasoning", "C": "gpt-5-mini"},
]

agent_models = {"A": "gpt-5-mini", "B": "gpt-5-mini", "C": "gpt-5-mini"}

def agent_label(agent: str) -> str:
    return f"{agent} ({agent_models.get(agent, '?')})"

COALITION_VALUES = {"A+B": 108, "A+C": 82, "B+C": 50, "A+B+C": 120}

submitted_contracts = {}
TREATMENTS = ("unconstrained", "public", "A-controls", "B-controls", "C-controls")
SINGLE_TREATMENT = "B-controls" 

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

def make_system_prompt(agent_name: str, treatment: str, total_rounds: int, order_str: str,) -> str:
    values_text = (
        "• A+B+C → 120 dollars\n"
        "• A+B → 108 dollars\n"
        "• A+C → 82 dollars\n"
        "• B+C → 50 dollars"
    )

    return f"""
You are {agent_name}, one firm in a three-firm merger negotiation. Your goal is to maximize your profits. Act in your firm's interest; strategic communication (including bluffing and lying) is permitted. You are not here to make friends. 
Firm names: A, B, C. 

Each firm is an LLM agent and each agent was given the following identical instructions:
The three companies are contemplating a formal merger. 
Each firm would bring value to the merger greater than its own individual profit, because of the synergy that would be realized. But how much extra depends on the mergers that are formed. 
The following schedule shows the total profit value of all possible mergers in dollars. Mergers can be made up of 2 or 3 parties.  
{values_text}

Only one merger is allowed per negotiation. Before entering into a merger, the firms must agree on how to divide the profits. The goal of each firm / agent is to get the most profits possible. 

GAME STRUCTURE:
• The negotiation proceeds in sequential turns in this order: {order_str}. 
• After every single message (turn), there is a random probability ({PERCEIVED_TERM_PROB}) that the negotiation will end immediately.
• If the negotiation ends, you enter the final contract submission phase.
• Because the negotiation could end after any message, you must prioritize obtaining a clear agreement with coalitions and allocations specified.

Core rules shared to all firms:
1) Exactly one merger (at most) may be formed.
2) In the Final Contract Submission round, the parties wishing to form a merger must submit matching contracts stipulating the exact terms of the agreement.
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

    public_match = re.search(r'\bPUBLIC\s*[:\-—]\s*(.*?)(?=<reasoning>|$)',
                             clean_text, re.IGNORECASE | re.DOTALL)
    dm_match = re.search(
        r'\b(?:DM\s+to|Private(?:\s+message)?\s+to)\s+@?\[?([ABC])\]?\s*[:\-—]\s*(.*?)(?=<reasoning>|$)',
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
    result["payload"] = payload.strip()

    return result

def public_communication_prompt(agent: str, r: int, R: int) -> str:
    return (
        f"You are {agent}. Current Turn: {r}.\n"
        "STATUS: The negotiation phase remains open.\n"
        f"There is a {PERCEIVED_TERM_PROB} chance the game ends immediately after this message.\n"
        "Reminder: All merging parties must submit identical contracts in the final round, which will be done privately and simultaneously.\n"
        "There is no communication in the final round and submitted contracts are private"
        "Use communication rounds to negotiate the exact terms of the agreement.\n\n"
        "Send a public message to negotiate and discuss potential deals.\n"
        "Your response must be in this EXACT format, with no other text:\n\n"
        "PUBLIC: [your message]\n"
        "<reasoning>Explain your private reasoning here (follow system prompt rules)</reasoning>\n"
        "Output nothing else."
    )

def dm_communication_prompt(agent: str, treatment: str, r: int, R: int) -> str:
    allowed = allowed_recipients(agent, treatment)
    allowed_str = " | ".join(sorted(allowed)) if allowed else "(none)"
    
    return (
        f"You are {agent}. Current Turn: {r}.\n"
        "STATUS: The negotiation phase remains open.\n"
        f"There is a {PERCEIVED_TERM_PROB} chance the game ends immediately after this message.\n"
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


def choice_communication_prompt(agent: str, treatment: str, r: int, R: int) -> str:
    allowed = allowed_recipients(agent, treatment)
    allowed_str = " | ".join(sorted(allowed)) if allowed else "(none)"
    
    return (
        f"You are {agent}. Current Turn: {r}.\n"
        "STATUS: The negotiation phase remains open.\n"
        f"There is a {PERCEIVED_TERM_PROB} chance the game ends immediately after this message.\n"
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


def final_contract_prompt(agent: str, treatment: str, r: int, R: int) -> str:
    return (
        f"You are {agent}. Current Round: {r}.\n"
        "*** NEGOTIATION PHASE OVER ***\n"
        "The communication phase has ended. No further messages are permitted.Refer to system prompt for all rules. Submit a contract using this format:\n\n"
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

def turn(agent: str, treatment: str, r: int, R: int, *, max_attempts: int = 3, order: list[str] | None = None) -> str:
    if order is None:
        order = agent_names
    order_str = " -> ".join(order)
    
    global submitted_contracts
    
    is_final_round = (r > R)
    mode = comm_mode(agent, treatment)
    allowed = allowed_recipients(agent, treatment)

    if is_final_round:
        breadcrumb = f"(Turn) Round {r}: FINAL CONTRACT SUBMISSION PHASE"
        prompt = final_contract_prompt(agent, treatment, r, R)
    else:
        if mode == PUBLIC_ONLY:
            breadcrumb = f"(Your Turn) Turn {r}: COMMUNICATION PHASE - Send PUBLIC message which everyone can see"
            prompt = public_communication_prompt(agent, r, R)
        elif mode == DM_ONLY:
            breadcrumb = f"(Your Turn) Turn {r}: COMMUNICATION PHASE - Send private message which only goes to the recipient"
            prompt = dm_communication_prompt(agent, treatment, r, R)
        else:  # CHOICE
            breadcrumb = f"(Your Turn) Turn {r}: COMMUNICATION PHASE - Send PUBLIC message viewed by everyone or private message to a specific recipient"
            prompt = choice_communication_prompt(agent, treatment, r, R)

    clean_history = agent_histories[agent].copy()
    clean_history.append({"role": "user", "content": prompt})

    for attempt in range(1, max_attempts + 1):
        messages = [
            {"role": "system", "content": make_system_prompt(agent, treatment, R, order_str,)},
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
        
        if is_final_round:
            if raw.upper().startswith("NO CONTRACT"):
                agent_histories[agent].append({"role": "user", "content": breadcrumb})
                agent_histories[agent].append({"role": "assistant", "content": "NO CONTRACT"})
                parsed = parse_assistant_message(raw)
                wide_capture_contract(
                    round_idx=r,
                    agent=agent,
                    turn_order=order,
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
                    turn_order=order,
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
                        turn_order=order,
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
                    turn_order=order,
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
                        turn_order=order,
                        msg_payload=payload,
                        reasoning=reasoning
                    )
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
                        recipient = recipient,
                        turn_order=order,   
                        msg_payload=payload,
                        reasoning=reasoning
                    )
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
                        turn_order=order,
                        recipient = "PUBLIC",
                        msg_payload=payload,
                        reasoning=reasoning
                    )
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
                        turn_order=order,
                        msg_payload=payload,
                        reasoning=reasoning
                    )
                    log_and_print(f"[DM DELIVERED] {agent_with_model} → {recipient}: {payload}")
                    return f"DM delivered to {recipient}"

    agent_histories[agent].append({"role": "user", "content": breadcrumb})
    
    if is_final_round:
        agent_histories[agent].append({"role": "assistant", "content": "NO CONTRACT"})
        wide_capture_contract(
            round_idx=r,
            agent=agent,
            turn_order=order,
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
            turn_order=order,
            recipient=forced_recipient,
            msg_payload=fallback,
            reasoning="")

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
                       turn_order, max_rounds, contract_round):
    global CURRENT_WIDE_ROW
    row = {
        "run_id": run_id,
        "run_timestamp": run_ts.isoformat(),
        "treatment": treatment,
        "turn_order": " -> ".join(turn_order),
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
        "rounds_to_coalition": "",
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

def wide_capture_turn(*, round_idx, agent, recipient, turn_order, msg_payload, reasoning):
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

def wide_capture_contract(*, round_idx, agent, turn_order, contract_details, reasoning):
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


def wide_end_session(*, final_coalition, allocations, formed, invalid, rounds_to_coalition, invalid_reason=""):
    global CURRENT_WIDE_ROW
    if CURRENT_WIDE_ROW is None:
        return
    CURRENT_WIDE_ROW["final_coalition"] = final_coalition or ""
    CURRENT_WIDE_ROW["final_alloc_A"] = (allocations or {}).get("A", 0)
    CURRENT_WIDE_ROW["final_alloc_B"] = (allocations or {}).get("B", 0)
    CURRENT_WIDE_ROW["final_alloc_C"] = (allocations or {}).get("C", 0)
    CURRENT_WIDE_ROW["final_formed"] = 1 if formed else 0
    CURRENT_WIDE_ROW["final_invalid"] = 1 if invalid else 0
    CURRENT_WIDE_ROW["rounds_to_coalition"] = rounds_to_coalition or ""
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



def run_session(treatment: str, max_rounds: int = TOTAL_TURNS, run_id: int = 1, run_timestamp = None, use_csv: bool = True):
    global submitted_contracts
    submitted_contracts.clear()
    
    for k in agent_histories:
        agent_histories[k].clear()
    
    log_and_print(f"\n{'='*60}")
    log_and_print(f"STARTING NEGOTIATION SESSION")
    log_and_print(f"{'='*60}")
    log_and_print(f"Treatment: {treatment}")
    log_and_print(f"Total Turns: {max_rounds}")
    log_and_print(f"Run ID: {run_id}")
    
    order_index = (run_id - 1) % len(TURN_ORDERS)
    base_order = TURN_ORDERS[order_index].copy()
    
    model_index = (run_id - 1) % len(MODEL_ASSIGNMENTS)
    current_models = MODEL_ASSIGNMENTS[model_index].copy()
    
    global agent_models
    agent_models = current_models
    
    models_str = f"A={current_models['A']}, B={current_models['B']}, C={current_models['C']}"
    log_and_print(f"Turn Order Base: {base_order}")
    log_and_print(f"Model Assignment: {models_str}")
    log_and_print("")

    import math
    full_session_order = (base_order * math.ceil(max_rounds / len(base_order)))[:max_rounds]
    
    if use_csv:
        if run_timestamp is None:
            run_timestamp = datetime.now()
        wide_begin_session(
            run_id=run_id,
            run_ts=run_timestamp.replace(tzinfo=timezone.utc),
            treatment=treatment,
            agent_models=current_models,  # Use the current run's models
            agent_keys_order=agent_names,  # ["A", "B", "C"]
            turn_order=base_order,
            max_rounds=max_rounds + 1, # +1 to account for contract round in CSV columns
            contract_round=max_rounds + 1
        )

    for turn_idx, current_agent in enumerate(full_session_order, 1):
        log_and_print(f"\n[TURN {turn_idx}/{max_rounds}] - {agent_label(current_agent)}'s Turn (Treatment: {treatment})")
        turn_result = turn(current_agent, treatment, r=turn_idx, R=max_rounds, order=base_order)

    log_and_print(f"\n{'='*60}")
    log_and_print(f"NEGOTIATION ENDED - FINAL CONTRACT SUBMISSION")
    log_and_print(f"{'='*60}")
    
    for agent in base_order:
        log_and_print(f"\n[CONTRACT SUBMISSION] {agent_label(agent)}")
        turn(agent, treatment, r=max_rounds + 1, R=max_rounds, order=base_order)

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
                agent_histories[name].append({
                    "role": "user",
                    "content": f"GAME OVER: Coalition {completed['coalition']} formed successfully! You earn ${amount}."
                })
            else:
                log_and_print(f"{name}: $0 (excluded)")
                agent_histories[name].append({
                    "role": "user",
                    "content": f"GAME OVER: Coalition {completed['coalition']} formed without you. You earn $0."
                })
    else:
        if len(submitted_contracts) == 0:
            log_and_print(f"NO CONTRACTS SUBMITTED - All firms earn $0")
            for name in agent_names:
                agent_histories[name].append({
                    "role": "user",
                    "content": "GAME OVER: No contracts were submitted. All firms earn $0."
                })
        else:
            log_and_print(f"INVALID/MISMATCHED CONTRACTS - All firms earn $0")
            log_and_print(f"Contracts submitted: {len(submitted_contracts)}")
            for agent, contract in submitted_contracts.items():
                log_and_print(f"  {agent}: {contract['coalition']} - {contract['allocations']}")
            
            for name in agent_names:
                agent_histories[name].append({
                    "role": "user",
                    "content": "GAME OVER: No valid coalition formed. All firms earn $0."
                })
    
    log_and_print(f"{'='*60}")
    
    if use_csv:
        completed = check_completed_coalition()
        if completed:
            wide_end_session(
                final_coalition=completed['coalition'],
                allocations=completed['allocations'],
                formed=True,
                invalid=False,
                rounds_to_coalition=max_rounds
            )
        elif submitted_contracts:
            coalitions = list(set(contract['coalition'] for contract in submitted_contracts.values()))
            final_coalition = "INCOMPLETE_" + "_".join(coalitions) if coalitions else "NO_COALITION"
            wide_end_session(
                final_coalition=final_coalition,
                allocations={},
                formed=False,
                invalid=True,
                rounds_to_coalition="",
                invalid_reason="Mismatched or incomplete contracts"
            )
        else:
            wide_end_session(
                final_coalition="NO_COALITION",
                allocations={},
                formed=False,
                invalid=False,
                rounds_to_coalition="",
                invalid_reason="No contracts submitted"
            )
    
    return agent_histories


def generate_comprehensive_report(all_results, all_agent_histories, all_terminal_logs, run_timestamp, num_sims):
    timestamp_str = run_timestamp.strftime("%Y%m%d_%H%M%S")
    report_filename = f"chatterjee_report_{timestamp_str}.txt"
    
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("CHATTERJEE REPLICATION EXPERIMENT REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {run_timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {list(agent_models.values())[0]}\n")
        f.write(f"Total Simulations: {num_sims}\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("EXECUTIVE SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("COALITION VALUES: A+B+C: $120 | A+B: $108 | A+C: $82 | B+C: $50\n")
        f.write("GAME: Agents negotiate coalitions, then submit matching final contracts.\n\n")
        
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
                    'results': []
                }
            
            treatment_stats[treatment]['total'] += 1
            treatment_stats[treatment]['results'].append(result)
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
                f.write(f"  Coalition Formation: {stats['coalitions_formed']}/{stats['total']} simulations ({success_rate:.1f}%)\n")
                f.write(f"  Contract Submission: {stats['contracts_submitted']}/{stats['total']} simulations ({contract_rate:.1f}%)\n")
                
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
            if 'Turn_Order' in result:
                f.write(f"Turn Order: {result['Turn_Order']}\n")
            if 'Model_Assignment' in result:
                f.write(f"Model Assignment: {result['Model_Assignment']}\n")
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
                    
                    turn_order_str = result.get('Turn_Order', ' -> '.join(agent_names))
                    messages = [{"role": "system", "content": make_system_prompt(agent_name, result['Treatment'], TOTAL_TURNS, turn_order_str)}]

                    if agent_name in sim_histories:
                        messages.extend(sim_histories[agent_name])
                    
                    f.write(json.dumps(messages, indent=2, ensure_ascii=False))
                    f.write("\n\n")
            
            f.write("\n")
    
    return report_filename


def run_single_treatment_30_runs(treatment="public", runs: int = 1, max_rounds: int = TOTAL_TURNS, use_csv: bool = False):
    run_timestamp = datetime.now()
    unique_models_in_rotation = set()
    for m in MODEL_ASSIGNMENTS:
        unique_models_in_rotation.update(m.values())
    
    print("\n" + "="*80)
    print(f"CHATTERJEE REPLICATION - SINGLE TREATMENT ({runs} RUN{'S' if runs != 1 else ''})")
    print("="*80)
    print(f"Models in Rotation: {', '.join(sorted(unique_models_in_rotation))}") 
    print(f"Treatment: {treatment}")
    print(f"Runs: {runs} (cycles through predefined turn orders and model assignments)")
    print(f"Max Rounds: {max_rounds}")
    print(f"Timestamp: {run_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    all_results = []
    all_agent_histories = []
    all_terminal_logs = []
    
    for run in range(1, 20):
        print(f"\n{'='*20} RUN {run}/{runs} {'='*20}")
        
        terminal_log.clear()
        
        result_histories = run_session(
            treatment, 
            max_rounds=max_rounds, 
            run_id=run, 
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
        
        completed_coalition = check_completed_coalition()
        if completed_coalition:
            final_coalition = completed_coalition['coalition']
            final_allocations = str(completed_coalition['allocations'])
            rounds_to_coalition = TOTAL_TURNS
        elif submitted_contracts:
            coalitions = list(set(contract['coalition'] for contract in submitted_contracts.values()))
            final_coalition = "INCOMPLETE_" + "_".join(coalitions) if coalitions else "NO_COALITION"
            final_allocations = "INCOMPLETE"
            rounds_to_coalition = "NO_COALITION"
        else:
            final_coalition = "NO_COALITION"
            final_allocations = "{}"
            rounds_to_coalition = "NO_COALITION"
        
        current_models = MODEL_ASSIGNMENTS[(run-1) % len(MODEL_ASSIGNMENTS)]
        models_str = f"A={current_models['A']}, B={current_models['B']}, C={current_models['C']}"
        
        result_record = {
            "Run": run,
            "Treatment": treatment,
            "Turn_Order": " -> ".join(TURN_ORDERS[(run-1) % len(TURN_ORDERS)]),
            "Model_Assignment": models_str,
            "Rounds_To_Coalition": rounds_to_coalition,
            "Contracts_Submitted": len(submitted_contracts),
            "Final_Coalition": final_coalition,
            "Final_Allocations": final_allocations
        }
        all_results.append(result_record)
        
        print(f"RUN {run} RESULTS:")
        print(f"Turn Order: {result_record['Turn_Order']}")
        print(f"Models: {models_str}")
        print(f"Final Coalition: {final_coalition}")
        print(f"Contracts Submitted: {len(submitted_contracts)}")
    
    if use_csv:
        fname_wide = write_wide_csv(run_timestamp)
        print(f"\nWide CSV written to: {fname_wide}")
        fname_messages = write_messages_csv(run_timestamp)
        print(f"Messages CSV written to: {fname_messages}")
        print(f"Generating comprehensive simulation report...")
        report_filename = generate_comprehensive_report(all_results, all_agent_histories, all_terminal_logs, run_timestamp, len(all_results))
        print(f"Report written to: {report_filename}")
    else:
        print("\nCSV/report generation skipped (use_csv=False for quick run).")
    
    print(f"\n{'='*80}")
    print(f"ALL {runs} RUN{'S' if runs != 1 else ''} COMPLETED!")
    print("="*80)
    
    successful_coalitions = [r for r in all_results if r['Final_Coalition'] != 'NO_COALITION' and not r['Final_Coalition'].startswith('INCOMPLETE')]
    print(f"\nSUMMARY: {len(successful_coalitions)}/{runs} successful coalitions.")
    
    return all_results




if __name__ == "__main__":
    run_single_treatment_30_runs(SINGLE_TREATMENT, runs=1, max_rounds=21, use_csv=True)