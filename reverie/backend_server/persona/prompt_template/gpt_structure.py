"""
Author: Joon Sung Park (joonspk@stanford.edu)

File: gpt_structure.py
Description: Wrapper functions for calling OpenAI APIs.
"""
import json
import random
import openai
import os
import time
import concurrent.futures as _futures

from utils import *

try:
  import telemetry_log
except Exception:
  telemetry_log = None

openai.api_key = openai_api_key
openai.api_base = llm_api_base
openai.proxy = apply_network_proxy()

# Layer 1: wall-clock timeout enforced by a worker thread, independent of
# openai 0.27's `request_timeout`. The library's timeout misses pooled-stale
# sockets when the proxy half-closes (observed 2026-05-29: 5 sims hung 30+ min
# on reads from urllib3-pooled connections after 127.0.0.1:7890 hiccuped).
# We hard-cap each create() at GA_LLM_WALL_TIMEOUT seconds (default 90).
_WALL_TIMEOUT = float(os.environ.get("GA_LLM_WALL_TIMEOUT", "90"))
_PARALLEL_PREPARE_WORKERS = int(os.environ.get("GA_PARALLEL_MAX_WORKERS", "4"))
_DEFAULT_WALL_WORKERS = (
    max(4, 2 * _PARALLEL_PREPARE_WORKERS)
    if os.environ.get("GA_PARALLEL_LLM", "0").strip() == "1"
    else 4)
_WALL_WORKERS = max(
    1, int(os.environ.get("GA_LLM_WALL_WORKERS", str(_DEFAULT_WALL_WORKERS))))
_WALL_EXECUTOR = _futures.ThreadPoolExecutor(max_workers=_WALL_WORKERS,
                                             thread_name_prefix="llm-wall")


def _call_with_wall_timeout(fn, *args, **kwargs):
  """Run `fn(*args, **kwargs)` in a worker, raise TimeoutError after
  _WALL_TIMEOUT seconds. The hung worker may leak (we cannot cancel a blocked
  socket read), but the caller is unblocked and can retry."""
  fut = _WALL_EXECUTOR.submit(fn, *args, **kwargs)
  try:
    return fut.result(timeout=_WALL_TIMEOUT)
  except _futures.TimeoutError:
    raise TimeoutError(
        f"openai call exceeded wall timeout {_WALL_TIMEOUT}s")

_LEGACY_COMPLETION_ENGINES = ("text-davinci-002", "text-davinci-003")
_API_ERROR_PREFIXES = ("GPT REQUEST ERROR", "ChatGPT ERROR")


# Module-level "last successful LLM call" timestamp. Read by the reverie
# step-watchdog (Layer 2) so it treats any successful API roundtrip as
# evidence of progress -- otherwise frame-0 (which has many LLM calls but
# no step increment for ~15-25 min wall-clock) trips the watchdog falsely.
_LAST_LLM_OK_TS = [time.time()]


def get_last_llm_ok_ts():
  return _LAST_LLM_OK_TS[0]


def _mark_llm_progress():
  _LAST_LLM_OK_TS[0] = time.time()


def _meter(call_type, model, response, latency_s=None):
  """Record token usage from an API response. Side-effect-free; never raises."""
  _mark_llm_progress()
  if telemetry_log is None:
    return
  try:
    if hasattr(response, "get"):
      usage = response.get("usage")
    else:
      usage = getattr(response, "usage", None)
  except Exception:
    usage = None
  telemetry_log.record_llm_call(call_type, model, usage, latency_s=latency_s)


def _meter_error(call_type, model, error, latency_s=None):
  """Record ops-only API failures without changing exception behavior."""
  if telemetry_log is not None:
    telemetry_log.record_llm_error(
        call_type, model, error, latency_s=latency_s)


def _print_api_error(label, error):
  print(f"{label} ({type(error).__name__}): {error}")


def _is_api_error_response(response):
  return (isinstance(response, str)
          and response.startswith(_API_ERROR_PREFIXES))


def is_api_error_response(response):
  return _is_api_error_response(response)


def _chat_completion_create(model, messages, **kwargs):
  kwargs.update({
    "model": model,
    "messages": messages,
    "request_timeout": llm_request_timeout,
  })
  headers = get_openrouter_headers()
  if headers:
    kwargs["headers"] = headers
  started = time.perf_counter()
  try:
    response = _call_with_wall_timeout(openai.ChatCompletion.create, **kwargs)
  except Exception as error:
    _meter_error("chat", model, error, time.perf_counter() - started)
    raise
  _meter("chat", model, response, time.perf_counter() - started)
  return response


def _completion_create(**kwargs):
  kwargs.setdefault("request_timeout", llm_request_timeout)
  headers = get_openrouter_headers()
  if headers:
    kwargs["headers"] = headers
  started = time.perf_counter()
  try:
    response = _call_with_wall_timeout(openai.Completion.create, **kwargs)
  except Exception as error:
    _meter_error("completion", kwargs.get("model"), error,
                 time.perf_counter() - started)
    raise
  _meter("completion", kwargs.get("model"), response,
         time.perf_counter() - started)
  return response


def _embedding_create(input, model):
  kwargs = {"input": input, "model": model, "request_timeout": llm_request_timeout}
  headers = get_openrouter_headers()
  if headers:
    kwargs["headers"] = headers
  started = time.perf_counter()
  try:
    response = _call_with_wall_timeout(openai.Embedding.create, **kwargs)
  except Exception as error:
    _meter_error("embedding", model, error, time.perf_counter() - started)
    raise
  _meter("embedding", model, response, time.perf_counter() - started)
  return response


def _should_use_chat_completion(model):
  return llm_provider == "openrouter" or "/" in model or model.startswith("gpt-")


def _chat_request_from_completion_prompt(prompt, model, gpt_parameter):
  kwargs = {
    "temperature": gpt_parameter["temperature"],
    "max_tokens": gpt_parameter["max_tokens"],
    "top_p": gpt_parameter["top_p"],
    "frequency_penalty": gpt_parameter["frequency_penalty"],
    "presence_penalty": gpt_parameter["presence_penalty"],
    "stop": gpt_parameter["stop"],
  }
  kwargs = {key: value for key, value in kwargs.items() if value is not None}
  completion = _chat_completion_create(
    model=model,
    messages=[{"role": "user", "content": prompt}],
    **kwargs,
  )
  return completion["choices"][0]["message"]["content"]

def temp_sleep(seconds=0.1):
  time.sleep(seconds)

def ChatGPT_single_request(prompt): 
  temp_sleep()

  completion = _chat_completion_create(
    model=llm_chat_model,
    messages=[{"role": "user", "content": prompt}],
  )
  return completion["choices"][0]["message"]["content"]


# ============================================================================
# #####################[SECTION 1: CHATGPT-3 STRUCTURE] ######################
# ============================================================================

def GPT4_request(prompt): 
  """
  Given a prompt and a dictionary of GPT parameters, make a request to OpenAI
  server and returns the response. 
  ARGS:
    prompt: a str prompt
    gpt_parameter: a python dictionary with the keys indicating the names of  
                   the parameter and the values indicating the parameter 
                   values.   
  RETURNS: 
    a str of GPT-3's response. 
  """
  temp_sleep()

  try: 
    completion = _chat_completion_create(
      model=llm_gpt4_model,
      messages=[{"role": "user", "content": prompt}],
    )
    return completion["choices"][0]["message"]["content"]
  
  except Exception as e: 
    _print_api_error("ChatGPT ERROR", e)
    return "ChatGPT ERROR"


def ChatGPT_request(prompt): 
  """
  Given a prompt and a dictionary of GPT parameters, make a request to OpenAI
  server and returns the response. 
  ARGS:
    prompt: a str prompt
    gpt_parameter: a python dictionary with the keys indicating the names of  
                   the parameter and the values indicating the parameter 
                   values.   
  RETURNS: 
    a str of GPT-3's response. 
  """
  # temp_sleep()
  try: 
    completion = _chat_completion_create(
      model=llm_chat_model,
      messages=[{"role": "user", "content": prompt}],
    )
    return completion["choices"][0]["message"]["content"]
  
  except Exception as e: 
    _print_api_error("ChatGPT ERROR", e)
    return "ChatGPT ERROR"


def GPT4_safe_generate_response(prompt, 
                                   example_output,
                                   special_instruction,
                                   repeat=3,
                                   fail_safe_response="error",
                                   func_validate=None,
                                   func_clean_up=None,
                                   verbose=False): 
  prompt = 'GPT-3 Prompt:\n"""\n' + prompt + '\n"""\n'
  prompt += f"Output the response to the prompt above in json. {special_instruction}\n"
  prompt += "Example output json:\n"
  prompt += '{"output": "' + str(example_output) + '"}'

  if verbose: 
    print ("CHAT GPT PROMPT")
    print (prompt)

  for i in range(repeat): 

    try: 
      curr_gpt_response = GPT4_request(prompt).strip()
      end_index = curr_gpt_response.rfind('}') + 1
      curr_gpt_response = curr_gpt_response[:end_index]
      curr_gpt_response = json.loads(curr_gpt_response)["output"]
      
      if func_validate(curr_gpt_response, prompt=prompt): 
        return func_clean_up(curr_gpt_response, prompt=prompt)
      
      if verbose: 
        print ("---- repeat count: \n", i, curr_gpt_response)
        print (curr_gpt_response)
        print ("~~~~")

    except: 
      pass

  return False


def ChatGPT_safe_generate_response(prompt, 
                                   example_output,
                                   special_instruction,
                                   repeat=3,
                                   fail_safe_response="error",
                                   func_validate=None,
                                   func_clean_up=None,
                                   verbose=False): 
  # prompt = 'GPT-3 Prompt:\n"""\n' + prompt + '\n"""\n'
  prompt = '"""\n' + prompt + '\n"""\n'
  prompt += f"Output the response to the prompt above in json. {special_instruction}\n"
  prompt += "Example output json:\n"
  prompt += '{"output": "' + str(example_output) + '"}'

  if verbose: 
    print ("CHAT GPT PROMPT")
    print (prompt)

  for i in range(repeat): 

    try: 
      curr_gpt_response = ChatGPT_request(prompt).strip()
      end_index = curr_gpt_response.rfind('}') + 1
      curr_gpt_response = curr_gpt_response[:end_index]
      curr_gpt_response = json.loads(curr_gpt_response)["output"]

      # print ("---ashdfaf")
      # print (curr_gpt_response)
      # print ("000asdfhia")
      
      if func_validate(curr_gpt_response, prompt=prompt): 
        return func_clean_up(curr_gpt_response, prompt=prompt)
      
      if verbose: 
        print ("---- repeat count: \n", i, curr_gpt_response)
        print (curr_gpt_response)
        print ("~~~~")

    except: 
      pass

  return False


def ChatGPT_safe_generate_response_OLD(prompt, 
                                   repeat=3,
                                   fail_safe_response="error",
                                   func_validate=None,
                                   func_clean_up=None,
                                   verbose=False): 
  if verbose: 
    print ("CHAT GPT PROMPT")
    print (prompt)

  for i in range(repeat): 
    try: 
      curr_gpt_response = ChatGPT_request(prompt).strip()
      if func_validate(curr_gpt_response, prompt=prompt): 
        return func_clean_up(curr_gpt_response, prompt=prompt)
      if verbose: 
        print (f"---- repeat count: {i}")
        print (curr_gpt_response)
        print ("~~~~")

    except: 
      pass
  print ("FAIL SAFE TRIGGERED") 
  return fail_safe_response


# ============================================================================
# ###################[SECTION 2: ORIGINAL GPT-3 STRUCTURE] ###################
# ============================================================================

def GPT_request(prompt, gpt_parameter): 
  """
  Given a prompt and a dictionary of GPT parameters, make a request to OpenAI
  server and returns the response. 
  ARGS:
    prompt: a str prompt
    gpt_parameter: a python dictionary with the keys indicating the names of  
                   the parameter and the values indicating the parameter 
                   values.   
  RETURNS: 
    a str of GPT-3's response. 
  """
  temp_sleep()
  try: 
    engine = gpt_parameter.get("engine", llm_completion_model)
    if engine in _LEGACY_COMPLETION_ENGINES:
      engine = llm_completion_model
    if _should_use_chat_completion(engine):
      return _chat_request_from_completion_prompt(prompt, engine, gpt_parameter)
    response = _completion_create(
                model=engine,
                prompt=prompt,
                temperature=gpt_parameter["temperature"],
                max_tokens=gpt_parameter["max_tokens"],
                top_p=gpt_parameter["top_p"],
                frequency_penalty=gpt_parameter["frequency_penalty"],
                presence_penalty=gpt_parameter["presence_penalty"],
                stream=gpt_parameter["stream"],
                stop=gpt_parameter["stop"],)
    return response.choices[0].text
  except Exception as e: 
    _print_api_error("GPT REQUEST ERROR", e)
    return "GPT REQUEST ERROR"


def generate_prompt(curr_input, prompt_lib_file): 
  """
  Takes in the current input (e.g. comment that you want to classifiy) and 
  the path to a prompt file. The prompt file contains the raw str prompt that
  will be used, which contains the following substr: !<INPUT>! -- this 
  function replaces this substr with the actual curr_input to produce the 
  final promopt that will be sent to the GPT3 server. 
  ARGS:
    curr_input: the input we want to feed in (IF THERE ARE MORE THAN ONE
                INPUT, THIS CAN BE A LIST.)
    prompt_lib_file: the path to the promopt file. 
  RETURNS: 
    a str prompt that will be sent to OpenAI's GPT server.  
  """
  if type(curr_input) == type("string"): 
    curr_input = [curr_input]
  curr_input = [str(i) for i in curr_input]

  f = open(prompt_lib_file, "r")
  prompt = f.read()
  f.close()
  for count, i in enumerate(curr_input):   
    prompt = prompt.replace(f"!<INPUT {count}>!", i)
  if "<commentblockmarker>###</commentblockmarker>" in prompt: 
    prompt = prompt.split("<commentblockmarker>###</commentblockmarker>")[1]
  return prompt.strip()


def safe_generate_response(prompt, 
                           gpt_parameter,
                           repeat=5,
                           fail_safe_response="error",
                           func_validate=None,
                           func_clean_up=None,
                           verbose=False): 
  if verbose: 
    print (prompt)

  for i in range(repeat): 
    curr_gpt_response = GPT_request(prompt, gpt_parameter)
    if _is_api_error_response(curr_gpt_response):
      if verbose or debug:
        print ("API ERROR FAIL SAFE: ", curr_gpt_response)
      return fail_safe_response
    try:
      if func_validate(curr_gpt_response, prompt=prompt): 
        return func_clean_up(curr_gpt_response, prompt=prompt)
    except Exception as e:
      if verbose or debug:
        _print_api_error("GPT VALIDATION ERROR", e)
    if verbose: 
      print ("---- repeat count: ", i, curr_gpt_response)
      print (curr_gpt_response)
      print ("~~~~")
  return fail_safe_response


def get_embedding(text, model=None):
  text = text.replace("\n", " ")
  if not text: 
    text = "this is blank"
  embedding_model = model or llm_embedding_model
  return _embedding_create(
          input=[text], model=embedding_model)['data'][0]['embedding']


if __name__ == '__main__':
  gpt_parameter = {"engine": "text-davinci-003", "max_tokens": 50, 
                   "temperature": 0, "top_p": 1, "stream": False,
                   "frequency_penalty": 0, "presence_penalty": 0, 
                   "stop": ['"']}
  curr_input = ["driving to a friend's house"]
  prompt_lib_file = "prompt_template/test_prompt_July5.txt"
  prompt = generate_prompt(curr_input, prompt_lib_file)

  def __func_validate(gpt_response): 
    if len(gpt_response.strip()) <= 1:
      return False
    if len(gpt_response.strip().split(" ")) > 1: 
      return False
    return True
  def __func_clean_up(gpt_response):
    cleaned_response = gpt_response.strip()
    return cleaned_response

  output = safe_generate_response(prompt, 
                                 gpt_parameter,
                                 5,
                                 "rest",
                                 __func_validate,
                                 __func_clean_up,
                                 True)

  print (output)




















