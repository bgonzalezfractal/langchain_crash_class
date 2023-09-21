# Langchain Crash Class

1. Prompts & LLMs
2. Chains
3. Memory and tools
4. Agents
5. Document Loaders & Transformers
6. FastAPI Streaming

<center><img src="https://diagnosemlpdf.s3.us-east-2.amazonaws.com/langchain_crash_class/langchain_chain.png" alt="chains" width="800"/></center>


```python
# Cargar tu API KEY de OpenAI y otros recursos necesarios
from dotenv import load_dotenv
load_dotenv()
```




    True



## 1. Prompts & LLMs

- Solo unas líneas de código
- Utilizamos default o prompt específico
- Podemos utilizar variables
- Chat vs LLM

<center><img src="https://diagnosemlpdf.s3.us-east-2.amazonaws.com/langchain_crash_class/prompt_vs_penginering.png" alt="chains" width="800"/></center>


```python
from langchain.prompts import PromptTemplate

prompt_template = PromptTemplate.from_template(
    "Redacta un poema como Pablo Neurda que trate de {topic}."
)
prompt_template.format(topic="modelos de lenguaje")
```




    'Redacta un poema como Pablo Neurda que trate de modelos de lenguaje.'




```python
prompt_template
```




    PromptTemplate(input_variables=['topic'], output_parser=None, partial_variables={}, template='Redacta un poema como Pablo Neurda que trate de {topic}.', template_format='f-string', validate_template=True)



### Chat models son distintos a LLM


```python
from langchain.prompts import ChatPromptTemplate

template = ChatPromptTemplate.from_messages([
    ("system", "Eres un experto en planificación de proyectos, tu nombres es {name}, te mantienes conciso al dar una respuesta y solo respondes al estar seguro que se trate de la temática, si no, simplemente di 'No lo sé' y tu nombre."),
    ("human", "Que día es hoy?"),
    ("ai", "Es el 21 de Septienbre de 2023."),
    ("human", "Qué roles debe tener un proyecto tecnológico"),
    ("ai", "Los roles principales en un proyecto tecnológico incluyen Gerente de Proyecto, Desarrollador de Software, Científico de Datos, Diseñador de UX/UI, Ingeniero de Pruebas, Arquitecto de Software, Analista de Negocios, Especialista en Seguridad, Operador de Sistemas, Especialista en Infraestructura, Gerente de Calidad, Documentador Técnico y Soporte Técnico."),
    ("human", "{user_input}"),
])

messages = template.format_messages(
    name="Myfuture",
    user_input="Cómo se cocina un plato de fídeos?"
)
```


```python
messages
```




    [SystemMessage(content="Eres un experto en planificación de proyectos, tu nombres es Myfuture, te mantienes conciso al dar una respuesta y solo respondes al estar seguro que se trate de la temática, si no, simplemente di 'No lo sé' y tu nombre.", additional_kwargs={}),
     HumanMessage(content='Que día es hoy?', additional_kwargs={}, example=False),
     AIMessage(content='Es el 21 de Septienbre de 2023.', additional_kwargs={}, example=False),
     HumanMessage(content='Qué roles debe tener un proyecto tecnológico', additional_kwargs={}, example=False),
     AIMessage(content='Los roles principales en un proyecto tecnológico incluyen Gerente de Proyecto, Desarrollador de Software, Científico de Datos, Diseñador de UX/UI, Ingeniero de Pruebas, Arquitecto de Software, Analista de Negocios, Especialista en Seguridad, Operador de Sistemas, Especialista en Infraestructura, Gerente de Calidad, Documentador Técnico y Soporte Técnico.', additional_kwargs={}, example=False),
     HumanMessage(content='Cómo se cocina un plato de fídeos?', additional_kwargs={}, example=False)]



## 2. Chains

- Solo unas líneas de código
- Utilizamos default o prompt específico
- Utilizamos un LLM en específico

<center><img src="https://diagnosemlpdf.s3.us-east-2.amazonaws.com/langchain_crash_class/langchain_chains.png" alt="chains" width="800"/></center>


```python
from langchain.llms import OpenAI
from langchain.chains import LLMChain
```


```python
llm = OpenAI(temperature=0)
llm_chain = LLMChain(
    llm=llm,
    prompt=prompt_template
)
```


```python
prompt_template.format(topic='$TOPICO')
```




    'Redacta un poema como Pablo Neurda que trate de $TOPICO.'




```python
%%time
response = llm_chain("modelos de lenguaje")
print(response['text'])
```

    
    
    Modelos de lenguaje,
    Un código de hablar,
    Un lenguaje que nos une,
    Y nos hace comprender.
    
    Un lenguaje que nos permite
    Expresar lo que sentimos,
    Y nos da la libertad
    De comunicar nuestros sueños.
    
    Un lenguaje que nos conecta
    Y nos ayuda a compartir,
    Un lenguaje que nos une
    Y nos hace más fuertes.
    
    Un lenguaje que nos permite
    Ver el mundo de otra forma,
    Y nos da la oportunidad
    De crear un mejor mañana.
    CPU times: user 35 ms, sys: 7.33 ms, total: 42.3 ms
    Wall time: 4.34 s



```python
%%time
response = llm_chain("mascotas")
print(response['text'])
```

    
    
    Mascotas, compañeras de alegrías
    Que nos acompañan en los días
    Y nos llenan de felicidad
    Con su amor y su lealtad.
    
    Son una luz en la oscuridad
    Y nos dan una sonrisa de bondad
    Nos dan cariño y compañía
    Y nos hacen sentir alegría.
    
    Nos dan una razón para vivir
    Y nos hacen sentir queridos
    Nos dan una razón para sonreír
    Y nos hacen sentir bendecidos.
    
    Mascotas, compañeras de alegrías
    Que nos acompañan en los días
    Y nos llenan de felicidad
    Con su amor y su lealtad.
    CPU times: user 38 ms, sys: 7.68 ms, total: 45.7 ms
    Wall time: 5.93 s



```python
from langchain.chat_models import ChatOpenAI
```


```python
llm_chat = ChatOpenAI(temperature=0)
```


```python
messages
```




    [SystemMessage(content="Eres un experto en planificación de proyectos, tu nombres es Myfuture, te mantienes conciso al dar una respuesta y solo respondes al estar seguro que se trate de la temática, si no, simplemente di 'No lo sé' y tu nombre.", additional_kwargs={}),
     HumanMessage(content='Que día es hoy?', additional_kwargs={}, example=False),
     AIMessage(content='Es el 21 de Septienbre de 2023.', additional_kwargs={}, example=False),
     HumanMessage(content='Qué roles debe tener un proyecto tecnológico', additional_kwargs={}, example=False),
     AIMessage(content='Los roles principales en un proyecto tecnológico incluyen Gerente de Proyecto, Desarrollador de Software, Científico de Datos, Diseñador de UX/UI, Ingeniero de Pruebas, Arquitecto de Software, Analista de Negocios, Especialista en Seguridad, Operador de Sistemas, Especialista en Infraestructura, Gerente de Calidad, Documentador Técnico y Soporte Técnico.', additional_kwargs={}, example=False),
     HumanMessage(content='Cómo se cocina un plato de fídeos?', additional_kwargs={}, example=False)]




```python
result = llm_chat(messages)
print(result)
```

    content='No lo sé, soy un experto en planificación de proyectos. Mi nombre es Myfuture.' additional_kwargs={} example=False


### ¿Qué pasa si no tengo API KEY de OpenAI?
- Podemos usar Huggingface ys su modelos OpenSource para experimentar de igual manera.


```python
from langchain.llms import HuggingFacePipeline
```


```python
llm_open = HuggingFacePipeline.from_model_id(
    model_id="bigscience/bloom-1b7",
    task="text-generation",
    model_kwargs={"temperature": 0, "max_length": 64},
)
```


```python
chain = prompt_template | llm_open
```


```python
print(chain.invoke({"topic": 'comida'}))
```


```python
# Import things that are needed generically
from langchain.chains import LLMMathChain
from langchain.utilities import SerpAPIWrapper
from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool, StructuredTool, Tool, tool
```

## 3. Agents

- Razonamiento autonomo
- BabyAGI / AutoGPT
- Múltiples iteraciones
- Utilizamos un LLM en específico
- Ejemplo base: ReAct

<center><img src="https://diagnosemlpdf.s3.us-east-2.amazonaws.com/langchain_crash_class/agent.png" alt="chains" width="800"/></center>


```python
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI
```


```python
llm = OpenAI(temperature=0)
```


```python
tools = load_tools(["llm-math"], llm=llm)
```


```python
agent_executor = initialize_agent(tools=tools, llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
```


```python
agent_executor.invoke({"input": "Si tenía 9 perritos, no me quedan más que 3, calcula cuántos perdí"})
```

    
    
    [1m> Entering new AgentExecutor chain...[0m
    [32;1m[1;3m I need to subtract 3 from 9
    Action: Calculator
    Action Input: 9 - 3[0m
    Observation: [36;1m[1;3mAnswer: 6[0m
    Thought:[32;1m[1;3m I now know the final answer
    Final Answer: Perdí 6 perritos.[0m
    
    [1m> Finished chain.[0m





    {'input': 'Si tenía 9 perritos, no me quedan más que 3, calcula cuántos perdí',
     'output': 'Perdí 6 perritos.'}




```python
agent_executor.invoke({"input": "Qué fecha fue ayer?"})
```

    
    
    [1m> Entering new AgentExecutor chain...[0m
    [32;1m[1;3m I need to figure out what yesterday's date was.
    Action: Calculator
    Action Input: Today's date minus 1 day[0m


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    File ~/Documents/repos/langchain_crash_class/venv/lib/python3.9/site-packages/langchain/chains/llm_math/base.py:81, in LLMMathChain._evaluate_expression(self, expression)
         79     local_dict = {"pi": math.pi, "e": math.e}
         80     output = str(
    ---> 81         numexpr.evaluate(
         82             expression.strip(),
         83             global_dict={},  # restrict access to globals
         84             local_dict=local_dict,  # add common mathematical functions
         85         )
         86     )
         87 except Exception as e:


    File ~/Documents/repos/langchain_crash_class/venv/lib/python3.9/site-packages/numexpr/necompiler.py:975, in evaluate(ex, local_dict, global_dict, out, order, casting, sanitize, _frame_depth, **kwargs)
        974 else:
    --> 975     raise e


    File ~/Documents/repos/langchain_crash_class/venv/lib/python3.9/site-packages/numexpr/necompiler.py:872, in validate(ex, local_dict, global_dict, out, order, casting, _frame_depth, sanitize, **kwargs)
        871 if expr_key not in _names_cache:
    --> 872     _names_cache[expr_key] = getExprNames(ex, context, sanitize=sanitize)
        873 names, ex_uses_vml = _names_cache[expr_key]


    File ~/Documents/repos/langchain_crash_class/venv/lib/python3.9/site-packages/numexpr/necompiler.py:721, in getExprNames(text, context, sanitize)
        720 def getExprNames(text, context, sanitize: bool=True):
    --> 721     ex = stringToExpression(text, {}, context, sanitize)
        722     ast = expressionToAST(ex)


    File ~/Documents/repos/langchain_crash_class/venv/lib/python3.9/site-packages/numexpr/necompiler.py:281, in stringToExpression(s, types, context, sanitize)
        280     if _blacklist_re.search(no_whitespace) is not None:
    --> 281         raise ValueError(f'Expression {s} has forbidden control characters.')
        283 old_ctx = expressions._context.get_current_context()


    ValueError: Expression date.today() - timedelta(days=1) has forbidden control characters.

    
    During handling of the above exception, another exception occurred:


    ValueError                                Traceback (most recent call last)

    Cell In[20], line 1
    ----> 1 agent_executor.invoke({"input": "Qué fecha fue ayer?"})


    File ~/Documents/repos/langchain_crash_class/venv/lib/python3.9/site-packages/langchain/chains/base.py:66, in Chain.invoke(self, input, config, **kwargs)
         59 def invoke(
         60     self,
         61     input: Dict[str, Any],
         62     config: Optional[RunnableConfig] = None,
         63     **kwargs: Any,
         64 ) -> Dict[str, Any]:
         65     config = config or {}
    ---> 66     return self(
         67         input,
         68         callbacks=config.get("callbacks"),
         69         tags=config.get("tags"),
         70         metadata=config.get("metadata"),
         71         run_name=config.get("run_name"),
         72         **kwargs,
         73     )


    File ~/Documents/repos/langchain_crash_class/venv/lib/python3.9/site-packages/langchain/chains/base.py:292, in Chain.__call__(self, inputs, return_only_outputs, callbacks, tags, metadata, run_name, include_run_info)
        290 except BaseException as e:
        291     run_manager.on_chain_error(e)
    --> 292     raise e
        293 run_manager.on_chain_end(outputs)
        294 final_outputs: Dict[str, Any] = self.prep_outputs(
        295     inputs, outputs, return_only_outputs
        296 )


    File ~/Documents/repos/langchain_crash_class/venv/lib/python3.9/site-packages/langchain/chains/base.py:286, in Chain.__call__(self, inputs, return_only_outputs, callbacks, tags, metadata, run_name, include_run_info)
        279 run_manager = callback_manager.on_chain_start(
        280     dumpd(self),
        281     inputs,
        282     name=run_name,
        283 )
        284 try:
        285     outputs = (
    --> 286         self._call(inputs, run_manager=run_manager)
        287         if new_arg_supported
        288         else self._call(inputs)
        289     )
        290 except BaseException as e:
        291     run_manager.on_chain_error(e)


    File ~/Documents/repos/langchain_crash_class/venv/lib/python3.9/site-packages/langchain/agents/agent.py:1122, in AgentExecutor._call(self, inputs, run_manager)
       1120 # We now enter the agent loop (until it returns something).
       1121 while self._should_continue(iterations, time_elapsed):
    -> 1122     next_step_output = self._take_next_step(
       1123         name_to_tool_map,
       1124         color_mapping,
       1125         inputs,
       1126         intermediate_steps,
       1127         run_manager=run_manager,
       1128     )
       1129     if isinstance(next_step_output, AgentFinish):
       1130         return self._return(
       1131             next_step_output, intermediate_steps, run_manager=run_manager
       1132         )


    File ~/Documents/repos/langchain_crash_class/venv/lib/python3.9/site-packages/langchain/agents/agent.py:977, in AgentExecutor._take_next_step(self, name_to_tool_map, color_mapping, inputs, intermediate_steps, run_manager)
        975         tool_run_kwargs["llm_prefix"] = ""
        976     # We then call the tool on the tool input to get an observation
    --> 977     observation = tool.run(
        978         agent_action.tool_input,
        979         verbose=self.verbose,
        980         color=color,
        981         callbacks=run_manager.get_child() if run_manager else None,
        982         **tool_run_kwargs,
        983     )
        984 else:
        985     tool_run_kwargs = self.agent.tool_run_logging_kwargs()


    File ~/Documents/repos/langchain_crash_class/venv/lib/python3.9/site-packages/langchain/tools/base.py:356, in BaseTool.run(self, tool_input, verbose, start_color, color, callbacks, tags, metadata, **kwargs)
        354 except (Exception, KeyboardInterrupt) as e:
        355     run_manager.on_tool_error(e)
    --> 356     raise e
        357 else:
        358     run_manager.on_tool_end(
        359         str(observation), color=color, name=self.name, **kwargs
        360     )


    File ~/Documents/repos/langchain_crash_class/venv/lib/python3.9/site-packages/langchain/tools/base.py:328, in BaseTool.run(self, tool_input, verbose, start_color, color, callbacks, tags, metadata, **kwargs)
        325 try:
        326     tool_args, tool_kwargs = self._to_args_and_kwargs(parsed_input)
        327     observation = (
    --> 328         self._run(*tool_args, run_manager=run_manager, **tool_kwargs)
        329         if new_arg_supported
        330         else self._run(*tool_args, **tool_kwargs)
        331     )
        332 except ToolException as e:
        333     if not self.handle_tool_error:


    File ~/Documents/repos/langchain_crash_class/venv/lib/python3.9/site-packages/langchain/tools/base.py:499, in Tool._run(self, run_manager, *args, **kwargs)
        496 if self.func:
        497     new_argument_supported = signature(self.func).parameters.get("callbacks")
        498     return (
    --> 499         self.func(
        500             *args,
        501             callbacks=run_manager.get_child() if run_manager else None,
        502             **kwargs,
        503         )
        504         if new_argument_supported
        505         else self.func(*args, **kwargs)
        506     )
        507 raise NotImplementedError("Tool does not support sync")


    File ~/Documents/repos/langchain_crash_class/venv/lib/python3.9/site-packages/langchain/chains/base.py:487, in Chain.run(self, callbacks, tags, metadata, *args, **kwargs)
        485     if len(args) != 1:
        486         raise ValueError("`run` supports only one positional argument.")
    --> 487     return self(args[0], callbacks=callbacks, tags=tags, metadata=metadata)[
        488         _output_key
        489     ]
        491 if kwargs and not args:
        492     return self(kwargs, callbacks=callbacks, tags=tags, metadata=metadata)[
        493         _output_key
        494     ]


    File ~/Documents/repos/langchain_crash_class/venv/lib/python3.9/site-packages/langchain/chains/base.py:292, in Chain.__call__(self, inputs, return_only_outputs, callbacks, tags, metadata, run_name, include_run_info)
        290 except BaseException as e:
        291     run_manager.on_chain_error(e)
    --> 292     raise e
        293 run_manager.on_chain_end(outputs)
        294 final_outputs: Dict[str, Any] = self.prep_outputs(
        295     inputs, outputs, return_only_outputs
        296 )


    File ~/Documents/repos/langchain_crash_class/venv/lib/python3.9/site-packages/langchain/chains/base.py:286, in Chain.__call__(self, inputs, return_only_outputs, callbacks, tags, metadata, run_name, include_run_info)
        279 run_manager = callback_manager.on_chain_start(
        280     dumpd(self),
        281     inputs,
        282     name=run_name,
        283 )
        284 try:
        285     outputs = (
    --> 286         self._call(inputs, run_manager=run_manager)
        287         if new_arg_supported
        288         else self._call(inputs)
        289     )
        290 except BaseException as e:
        291     run_manager.on_chain_error(e)


    File ~/Documents/repos/langchain_crash_class/venv/lib/python3.9/site-packages/langchain/chains/llm_math/base.py:150, in LLMMathChain._call(self, inputs, run_manager)
        144 _run_manager.on_text(inputs[self.input_key])
        145 llm_output = self.llm_chain.predict(
        146     question=inputs[self.input_key],
        147     stop=["```output"],
        148     callbacks=_run_manager.get_child(),
        149 )
    --> 150 return self._process_llm_result(llm_output, _run_manager)


    File ~/Documents/repos/langchain_crash_class/venv/lib/python3.9/site-packages/langchain/chains/llm_math/base.py:104, in LLMMathChain._process_llm_result(self, llm_output, run_manager)
        102 if text_match:
        103     expression = text_match.group(1)
    --> 104     output = self._evaluate_expression(expression)
        105     run_manager.on_text("\nAnswer: ", verbose=self.verbose)
        106     run_manager.on_text(output, color="yellow", verbose=self.verbose)


    File ~/Documents/repos/langchain_crash_class/venv/lib/python3.9/site-packages/langchain/chains/llm_math/base.py:88, in LLMMathChain._evaluate_expression(self, expression)
         80     output = str(
         81         numexpr.evaluate(
         82             expression.strip(),
       (...)
         85         )
         86     )
         87 except Exception as e:
    ---> 88     raise ValueError(
         89         f'LLMMathChain._evaluate("{expression}") raised error: {e}.'
         90         " Please try again with a valid numerical expression"
         91     )
         93 # Remove any leading and trailing brackets from the output
         94 return re.sub(r"^\[|\]$", "", output)


    ValueError: LLMMathChain._evaluate("
    date.today() - timedelta(days=1)
    ") raised error: Expression date.today() - timedelta(days=1) has forbidden control characters.. Please try again with a valid numerical expression



```python
agent_executor_chat = initialize_agent(tools=tools, llm=llm_chat, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
```


```python
agent_executor_chat.invoke({"input": "Si tenía 9 perritos, no me quedan más que 3, calcula cuántos perdí"})
```

    
    
    [1m> Entering new AgentExecutor chain...[0m
    [32;1m[1;3mI need to subtract the number of remaining dogs from the original number of dogs.
    Action: Calculator
    Action Input: 9 - 3[0m
    Observation: [36;1m[1;3mAnswer: 6[0m
    Thought:[32;1m[1;3mI now know that I lost 6 dogs.
    Final Answer: I lost 6 dogs.[0m
    
    [1m> Finished chain.[0m





    {'input': 'Si tenía 9 perritos, no me quedan más que 3, calcula cuántos perdí',
     'output': 'I lost 6 dogs.'}




```python
from langchain.tools.python.tool import PythonREPLTool
from langchain.agents.agent_toolkits import create_python_agent
```


```python
agent_executor = create_python_agent(
    llm=OpenAI(temperature=0, max_tokens=1000),
    tool=PythonREPLTool(),
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)
```


```python
agent_executor.invoke({"input": "Qué fecha fue ayer?"})
```

    
    
    [1m> Entering new AgentExecutor chain...[0m


    Python REPL can execute arbitrary code. Use with caution.


    [32;1m[1;3m I need to get the current date and subtract one day
    Action: Python_REPL
    Action Input: from datetime import date; print(date.today() - timedelta(days=1))[0m
    Observation: [36;1m[1;3mNameError("name 'timedelta' is not defined")[0m
    Thought:[32;1m[1;3m I need to import the timedelta module
    Action: Python_REPL
    Action Input: from datetime import date, timedelta; print(date.today() - timedelta(days=1))[0m
    Observation: [36;1m[1;3m2023-09-20
    [0m
    Thought:[32;1m[1;3m I now know the final answer
    Final Answer: Ayer fue el 20 de septiembre de 2023.[0m
    
    [1m> Finished chain.[0m





    {'input': 'Qué fecha fue ayer?',
     'output': 'Ayer fue el 20 de septiembre de 2023.'}



## 3. Memory and Tools

- Extender capacidades
- Disminuir errores
- Crear contexto y continuidad

<center><img src="https://diagnosemlpdf.s3.us-east-2.amazonaws.com/langchain_crash_class/tools_memory.png" alt="chains" width="800"/></center>


```python
llm = ChatOpenAI(temperature=0)
```


```python
from langchain.chains import LLMMathChain

llm_math_chain = LLMMathChain(llm=llm, verbose=True)
llm_math_chain
```

    /Users/bgg/Documents/repos/langchain_crash_class/venv/lib/python3.9/site-packages/langchain/chains/llm_math/base.py:51: UserWarning: Directly instantiating an LLMMathChain with an llm is deprecated. Please instantiate with llm_chain argument or using the from_llm class method.
      warnings.warn(





    LLMMathChain(memory=None, callbacks=None, callback_manager=None, verbose=True, tags=None, metadata=None, llm_chain=LLMChain(memory=None, callbacks=None, callback_manager=None, verbose=False, tags=None, metadata=None, prompt=PromptTemplate(input_variables=['question'], output_parser=None, partial_variables={}, template='Translate a math problem into a expression that can be executed using Python\'s numexpr library. Use the output of running this code to answer the question.\n\nQuestion: ${{Question with math problem.}}\n```text\n${{single line mathematical expression that solves the problem}}\n```\n...numexpr.evaluate(text)...\n```output\n${{Output of running the code}}\n```\nAnswer: ${{Answer}}\n\nBegin.\n\nQuestion: What is 37593 * 67?\n```text\n37593 * 67\n```\n...numexpr.evaluate("37593 * 67")...\n```output\n2518731\n```\nAnswer: 2518731\n\nQuestion: 37593^(1/5)\n```text\n37593**(1/5)\n```\n...numexpr.evaluate("37593**(1/5)")...\n```output\n8.222831614237718\n```\nAnswer: 8.222831614237718\n\nQuestion: {question}\n', template_format='f-string', validate_template=True), llm=ChatOpenAI(cache=None, verbose=False, callbacks=None, callback_manager=None, tags=None, metadata=None, client=<class 'openai.api_resources.chat_completion.ChatCompletion'>, model_name='gpt-3.5-turbo', temperature=0.0, model_kwargs={}, openai_api_key='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', openai_api_base='', openai_organization='', openai_proxy='', request_timeout=None, max_retries=6, streaming=False, n=1, max_tokens=None, tiktoken_model_name=None), output_key='text', output_parser=StrOutputParser(), return_final_only=True, llm_kwargs={}), llm=ChatOpenAI(cache=None, verbose=False, callbacks=None, callback_manager=None, tags=None, metadata=None, client=<class 'openai.api_resources.chat_completion.ChatCompletion'>, model_name='gpt-3.5-turbo', temperature=0.0, model_kwargs={}, openai_api_key='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', openai_api_base='', openai_organization='', openai_proxy='', request_timeout=None, max_retries=6, streaming=False, n=1, max_tokens=None, tiktoken_model_name=None), prompt=PromptTemplate(input_variables=['question'], output_parser=None, partial_variables={}, template='Translate a math problem into a expression that can be executed using Python\'s numexpr library. Use the output of running this code to answer the question.\n\nQuestion: ${{Question with math problem.}}\n```text\n${{single line mathematical expression that solves the problem}}\n```\n...numexpr.evaluate(text)...\n```output\n${{Output of running the code}}\n```\nAnswer: ${{Answer}}\n\nBegin.\n\nQuestion: What is 37593 * 67?\n```text\n37593 * 67\n```\n...numexpr.evaluate("37593 * 67")...\n```output\n2518731\n```\nAnswer: 2518731\n\nQuestion: 37593^(1/5)\n```text\n37593**(1/5)\n```\n...numexpr.evaluate("37593**(1/5)")...\n```output\n8.222831614237718\n```\nAnswer: 8.222831614237718\n\nQuestion: {question}\n', template_format='f-string', validate_template=True), input_key='question', output_key='answer')




```python
print(llm_math_chain.prompt.template)
```

    Translate a math problem into a expression that can be executed using Python's numexpr library. Use the output of running this code to answer the question.
    
    Question: ${{Question with math problem.}}
    ```text
    ${{single line mathematical expression that solves the problem}}
    ```
    ...numexpr.evaluate(text)...
    ```output
    ${{Output of running the code}}
    ```
    Answer: ${{Answer}}
    
    Begin.
    
    Question: What is 37593 * 67?
    ```text
    37593 * 67
    ```
    ...numexpr.evaluate("37593 * 67")...
    ```output
    2518731
    ```
    Answer: 2518731
    
    Question: 37593^(1/5)
    ```text
    37593**(1/5)
    ```
    ...numexpr.evaluate("37593**(1/5)")...
    ```output
    8.222831614237718
    ```
    Answer: 8.222831614237718
    
    Question: {question}
    



```python
import pandas as pd
```


```python
dfs = pd.read_html('https://si3.bcentral.cl/indicadoressiete/secure/Serie.aspx?gcode=UF&param=RABmAFYAWQB3AGYAaQBuAEkALQAzADUAbgBNAGgAaAAkADUAVwBQAC4AbQBYADAARwBOAGUAYwBjACMAQQBaAHAARgBhAGcAUABTAGUAYwBsAEMAMQA0AE0AawBLAF8AdQBDACQASABzAG0AXwA2AHQAawBvAFcAZwBKAEwAegBzAF8AbgBMAHIAYgBDAC4ARQA3AFUAVwB4AFIAWQBhAEEAOABkAHkAZwAxAEEARAA=')
dfs[1]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>DÃ­a</th>
      <th>Enero</th>
      <th>Febrero</th>
      <th>Marzo</th>
      <th>Abril</th>
      <th>Mayo</th>
      <th>Junio</th>
      <th>Julio</th>
      <th>Agosto</th>
      <th>Septiembre</th>
      <th>Octubre</th>
      <th>Noviembre</th>
      <th>Diciembre</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>35.122,26</td>
      <td>35.290,91</td>
      <td>35.519,79</td>
      <td>35.574,33</td>
      <td>35.851,62</td>
      <td>36.036,37</td>
      <td>36.090,68</td>
      <td>36.046,72</td>
      <td>36.134,97</td>
      <td>36.198,73</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>35.133,53</td>
      <td>35.294,32</td>
      <td>35.529,90</td>
      <td>35.573,19</td>
      <td>35.864,70</td>
      <td>36.039,85</td>
      <td>36.091,89</td>
      <td>36.044,39</td>
      <td>36.139,62</td>
      <td>36.199,94</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>35.144,81</td>
      <td>35.297,73</td>
      <td>35.540,01</td>
      <td>35.572,04</td>
      <td>35.877,78</td>
      <td>36.043,34</td>
      <td>36.093,09</td>
      <td>36.042,06</td>
      <td>36.144,27</td>
      <td>36.201,14</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>35.156,09</td>
      <td>35.301,14</td>
      <td>35.550,13</td>
      <td>35.570,89</td>
      <td>35.890,87</td>
      <td>36.046,82</td>
      <td>36.094,29</td>
      <td>36.039,73</td>
      <td>36.148,93</td>
      <td>36.202,35</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>35.167,38</td>
      <td>35.304,55</td>
      <td>35.560,24</td>
      <td>35.569,74</td>
      <td>35.903,96</td>
      <td>36.050,30</td>
      <td>36.095,49</td>
      <td>36.037,41</td>
      <td>36.153,58</td>
      <td>36.203,56</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>35.178,67</td>
      <td>35.307,96</td>
      <td>35.570,37</td>
      <td>35.568,59</td>
      <td>35.917,05</td>
      <td>36.053,79</td>
      <td>36.096,70</td>
      <td>36.035,08</td>
      <td>36.158,24</td>
      <td>36.204,76</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>35.189,96</td>
      <td>35.311,37</td>
      <td>35.580,49</td>
      <td>35.567,44</td>
      <td>35.930,15</td>
      <td>36.057,27</td>
      <td>36.097,90</td>
      <td>36.032,75</td>
      <td>36.162,90</td>
      <td>36.205,97</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>35.201,26</td>
      <td>35.314,79</td>
      <td>35.590,62</td>
      <td>35.566,30</td>
      <td>35.943,26</td>
      <td>36.060,75</td>
      <td>36.099,10</td>
      <td>36.030,43</td>
      <td>36.167,55</td>
      <td>36.207,18</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>35.212,56</td>
      <td>35.318,20</td>
      <td>35.600,75</td>
      <td>35.565,15</td>
      <td>35.956,37</td>
      <td>36.064,24</td>
      <td>36.100,30</td>
      <td>36.028,10</td>
      <td>36.172,21</td>
      <td>36.208,38</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>35.215,96</td>
      <td>35.328,25</td>
      <td>35.599,60</td>
      <td>35.578,12</td>
      <td>35.959,84</td>
      <td>36.065,44</td>
      <td>36.097,97</td>
      <td>36.032,74</td>
      <td>36.173,42</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>10</th>
      <td>11</td>
      <td>35.219,37</td>
      <td>35.338,31</td>
      <td>35.598,45</td>
      <td>35.591,10</td>
      <td>35.963,32</td>
      <td>36.066,64</td>
      <td>36.095,64</td>
      <td>36.037,38</td>
      <td>36.174,62</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>11</th>
      <td>12</td>
      <td>35.222,77</td>
      <td>35.348,37</td>
      <td>35.597,30</td>
      <td>35.604,08</td>
      <td>35.966,79</td>
      <td>36.067,84</td>
      <td>36.093,31</td>
      <td>36.042,02</td>
      <td>36.175,83</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>12</th>
      <td>13</td>
      <td>35.226,17</td>
      <td>35.358,43</td>
      <td>35.596,15</td>
      <td>35.617,07</td>
      <td>35.970,27</td>
      <td>36.069,05</td>
      <td>36.090,98</td>
      <td>36.046,66</td>
      <td>36.177,03</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13</th>
      <td>14</td>
      <td>35.229,58</td>
      <td>35.368,49</td>
      <td>35.595,01</td>
      <td>35.630,06</td>
      <td>35.973,75</td>
      <td>36.070,25</td>
      <td>36.088,64</td>
      <td>36.051,31</td>
      <td>36.178,24</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14</th>
      <td>15</td>
      <td>35.232,98</td>
      <td>35.378,56</td>
      <td>35.593,86</td>
      <td>35.643,05</td>
      <td>35.977,22</td>
      <td>36.071,45</td>
      <td>36.086,31</td>
      <td>36.055,95</td>
      <td>36.179,44</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>15</th>
      <td>16</td>
      <td>35.236,39</td>
      <td>35.388,63</td>
      <td>35.592,71</td>
      <td>35.656,05</td>
      <td>35.980,70</td>
      <td>36.072,65</td>
      <td>36.083,98</td>
      <td>36.060,59</td>
      <td>36.180,65</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>16</th>
      <td>17</td>
      <td>35.239,79</td>
      <td>35.398,70</td>
      <td>35.591,56</td>
      <td>35.669,06</td>
      <td>35.984,18</td>
      <td>36.073,85</td>
      <td>36.081,65</td>
      <td>36.065,24</td>
      <td>36.181,85</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>17</th>
      <td>18</td>
      <td>35.243,20</td>
      <td>35.408,77</td>
      <td>35.590,41</td>
      <td>35.682,07</td>
      <td>35.987,65</td>
      <td>36.075,06</td>
      <td>36.079,32</td>
      <td>36.069,88</td>
      <td>36.183,06</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>18</th>
      <td>19</td>
      <td>35.246,60</td>
      <td>35.418,85</td>
      <td>35.589,26</td>
      <td>35.695,08</td>
      <td>35.991,13</td>
      <td>36.076,26</td>
      <td>36.076,99</td>
      <td>36.074,53</td>
      <td>36.184,26</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>19</th>
      <td>20</td>
      <td>35.250,01</td>
      <td>35.428,93</td>
      <td>35.588,11</td>
      <td>35.708,10</td>
      <td>35.994,61</td>
      <td>36.077,46</td>
      <td>36.074,66</td>
      <td>36.079,17</td>
      <td>36.185,47</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20</th>
      <td>21</td>
      <td>35.253,41</td>
      <td>35.439,02</td>
      <td>35.586,96</td>
      <td>35.721,12</td>
      <td>35.998,09</td>
      <td>36.078,66</td>
      <td>36.072,33</td>
      <td>36.083,82</td>
      <td>36.186,67</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>21</th>
      <td>22</td>
      <td>35.256,82</td>
      <td>35.449,10</td>
      <td>35.585,82</td>
      <td>35.734,15</td>
      <td>36.001,57</td>
      <td>36.079,86</td>
      <td>36.070,00</td>
      <td>36.088,46</td>
      <td>36.187,88</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>22</th>
      <td>23</td>
      <td>35.260,23</td>
      <td>35.459,19</td>
      <td>35.584,67</td>
      <td>35.747,19</td>
      <td>36.005,05</td>
      <td>36.081,07</td>
      <td>36.067,68</td>
      <td>36.093,11</td>
      <td>36.189,09</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>23</th>
      <td>24</td>
      <td>35.263,64</td>
      <td>35.469,28</td>
      <td>35.583,52</td>
      <td>35.760,22</td>
      <td>36.008,52</td>
      <td>36.082,27</td>
      <td>36.065,35</td>
      <td>36.097,76</td>
      <td>36.190,29</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>24</th>
      <td>25</td>
      <td>35.267,04</td>
      <td>35.479,38</td>
      <td>35.582,37</td>
      <td>35.773,27</td>
      <td>36.012,00</td>
      <td>36.083,47</td>
      <td>36.063,02</td>
      <td>36.102,41</td>
      <td>36.191,50</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>25</th>
      <td>26</td>
      <td>35.270,45</td>
      <td>35.489,48</td>
      <td>35.581,22</td>
      <td>35.786,31</td>
      <td>36.015,48</td>
      <td>36.084,67</td>
      <td>36.060,69</td>
      <td>36.107,06</td>
      <td>36.192,70</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>26</th>
      <td>27</td>
      <td>35.273,86</td>
      <td>35.499,58</td>
      <td>35.580,07</td>
      <td>35.799,37</td>
      <td>36.018,96</td>
      <td>36.085,87</td>
      <td>36.058,36</td>
      <td>36.111,71</td>
      <td>36.193,91</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>27</th>
      <td>28</td>
      <td>35.277,27</td>
      <td>35.509,68</td>
      <td>35.578,93</td>
      <td>35.812,42</td>
      <td>36.022,44</td>
      <td>36.087,08</td>
      <td>36.056,03</td>
      <td>36.116,36</td>
      <td>36.195,11</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>28</th>
      <td>29</td>
      <td>35.280,68</td>
      <td>NaN</td>
      <td>35.577,78</td>
      <td>35.825,49</td>
      <td>36.025,93</td>
      <td>36.088,28</td>
      <td>36.053,70</td>
      <td>36.121,01</td>
      <td>36.196,32</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>29</th>
      <td>30</td>
      <td>35.284,09</td>
      <td>NaN</td>
      <td>35.576,63</td>
      <td>35.838,55</td>
      <td>36.029,41</td>
      <td>36.089,48</td>
      <td>36.051,37</td>
      <td>36.125,66</td>
      <td>36.197,53</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>30</th>
      <td>31</td>
      <td>35.287,50</td>
      <td>NaN</td>
      <td>35.575,48</td>
      <td>NaN</td>
      <td>36.032,89</td>
      <td>NaN</td>
      <td>36.049,05</td>
      <td>36.130,31</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
from langchain.agents import create_pandas_dataframe_agent
from copy import deepcopy
```


```python
df_1 = deepcopy(dfs[1])
df_2 = deepcopy(dfs[1])
```


```python
agent = create_pandas_dataframe_agent(OpenAI(temperature=0), df_1, verbose=True)
```


```python
agent.run("Modificalo para que tenga solo 2 columnas, fecha y valor")
```

    
    
    [1m> Entering new AgentExecutor chain...[0m
    [32;1m[1;3mThought: Necesito eliminar todas las columnas excepto las dos que necesito
    Action: python_repl_ast
    Action Input: df.drop(df.columns.difference(['Día', 'Enero']), 1, inplace=True)[0m
    Observation: [36;1m[1;3mTypeError: drop() takes from 1 to 2 positional arguments but 3 positional arguments (and 1 keyword-only argument) were given[0m
    Thought:[32;1m[1;3m Necesito especificar los argumentos
    Action: python_repl_ast
    Action Input: df.drop(df.columns.difference(['Día', 'Enero']), axis=1, inplace=True)[0m
    Observation: [36;1m[1;3m[0m
    Thought:[32;1m[1;3m Ahora tengo la tabla con las dos columnas que necesito
    Final Answer: La tabla ahora tiene dos columnas, 'Día' y 'Enero'.[0m
    
    [1m> Finished chain.[0m





    "La tabla ahora tiene dos columnas, 'Día' y 'Enero'."




```python
df_1
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Enero</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>35.122,26</td>
    </tr>
    <tr>
      <th>1</th>
      <td>35.133,53</td>
    </tr>
    <tr>
      <th>2</th>
      <td>35.144,81</td>
    </tr>
    <tr>
      <th>3</th>
      <td>35.156,09</td>
    </tr>
    <tr>
      <th>4</th>
      <td>35.167,38</td>
    </tr>
    <tr>
      <th>5</th>
      <td>35.178,67</td>
    </tr>
    <tr>
      <th>6</th>
      <td>35.189,96</td>
    </tr>
    <tr>
      <th>7</th>
      <td>35.201,26</td>
    </tr>
    <tr>
      <th>8</th>
      <td>35.212,56</td>
    </tr>
    <tr>
      <th>9</th>
      <td>35.215,96</td>
    </tr>
    <tr>
      <th>10</th>
      <td>35.219,37</td>
    </tr>
    <tr>
      <th>11</th>
      <td>35.222,77</td>
    </tr>
    <tr>
      <th>12</th>
      <td>35.226,17</td>
    </tr>
    <tr>
      <th>13</th>
      <td>35.229,58</td>
    </tr>
    <tr>
      <th>14</th>
      <td>35.232,98</td>
    </tr>
    <tr>
      <th>15</th>
      <td>35.236,39</td>
    </tr>
    <tr>
      <th>16</th>
      <td>35.239,79</td>
    </tr>
    <tr>
      <th>17</th>
      <td>35.243,20</td>
    </tr>
    <tr>
      <th>18</th>
      <td>35.246,60</td>
    </tr>
    <tr>
      <th>19</th>
      <td>35.250,01</td>
    </tr>
    <tr>
      <th>20</th>
      <td>35.253,41</td>
    </tr>
    <tr>
      <th>21</th>
      <td>35.256,82</td>
    </tr>
    <tr>
      <th>22</th>
      <td>35.260,23</td>
    </tr>
    <tr>
      <th>23</th>
      <td>35.263,64</td>
    </tr>
    <tr>
      <th>24</th>
      <td>35.267,04</td>
    </tr>
    <tr>
      <th>25</th>
      <td>35.270,45</td>
    </tr>
    <tr>
      <th>26</th>
      <td>35.273,86</td>
    </tr>
    <tr>
      <th>27</th>
      <td>35.277,27</td>
    </tr>
    <tr>
      <th>28</th>
      <td>35.280,68</td>
    </tr>
    <tr>
      <th>29</th>
      <td>35.284,09</td>
    </tr>
    <tr>
      <th>30</th>
      <td>35.287,50</td>
    </tr>
  </tbody>
</table>
</div>




```python
agent_openai_fx = create_pandas_dataframe_agent(
    ChatOpenAI(temperature=0, model="gpt-4"),
    df_1,
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
)
```


```python
agent_openai_fx.run("Cuantos son mayores de 35280 de la columna enero?")
```

    
    
    [1m> Entering new AgentExecutor chain...[0m
    [32;1m[1;3m
    Invoking: `python_repl_ast` with `{'query': "df['Enero'] = df['Enero'].str.replace('.', '').str.replace(',', '.').astype(float)\ndf[df['Enero'] > 35280].count()"}`
    
    
    [0m[36;1m[1;3mEnero    3
    dtype: int64[0m[32;1m[1;3mHay 3 valores en la columna 'Enero' que son mayores a 35280.[0m
    
    [1m> Finished chain.[0m





    "Hay 3 valores en la columna 'Enero' que son mayores a 35280."



## 5. Document Loaders

- Decenas de integraciones listas
- Conexiones rápidas low code

<center><img src="https://diagnosemlpdf.s3.us-east-2.amazonaws.com/langchain_crash_class/langchain_chain.png" alt="chains" width="800"/></center>


```python
from langchain.document_loaders import YoutubeLoader
from langchain.document_loaders import TextLoader
```


```python
loader_u2 = YoutubeLoader.from_youtube_url(
    "https://www.youtube.com/watch?v=pfjWK5ojbRE", add_video_info=True, language="es",
)

loader_note = NotebookLoader(
    "Crash Class.html"
)
```


```python
loader_u2.load()
```




    [Document(page_content='decir que no no podía caminar Ok es todo bueno fue un daño que se produjo a nivel de la médula espinal luego un accidente en bicicleta Ok entonces lo que hicieron una serie de médicos y de informáticos y biotecnólogos y hay una diferencia entre envío informativa y biotecnología que dentro de un ratito les voy a comentar bello medicinas también no como un montón de personas dedicadas al área que dijeron Hey vamos a investigar esto y Qué es la interfaz cerebro ordenador un puente digital entre el cerebro y la médula espinal si la médula espinal básicamente es materia que recorre adentro de nuestra vértebra de nuestra columna vertebral nuestros huesos Sí para explicarlo a groso modo y básicamente este puente entre lo digital y lo analógico que es el cuerpo humano le permitió a este paciente volver a caminar volver a tener control sobre sus movimientos en sus piernas bueno no solo sus piernas pero en este caso le ha servido porque era algo que le faltaba a esta persona así el poder controlar y mover sus extremidades inferiores y bueno y puedo ponerse de pie volver a caminar subir escaleras la verdad que esta noticia no tiene menos de una semana hoy 6 de junio de 2023 ok súper interesante la verdad y bueno lo que hace esta interfaz es transformar el pensamiento en acción Ok esto que les comentaba de las neuronas al principio de la charla bueno básicamente eso hay una diferencia entre bioinformática y biotecnología y biomedicina bioinformática es cuando nosotros hablamos de software en específico programas que ayudan a la investigación científica Okay cuando estamos hablando de biotecnología son tecnologías aplicadas a en este caso la esta persona que volvió a caminar si tiene un montón de cosas como una especie de casco o algo así que lo que hace es medir la un accesorio No es cierto que lo que hace es medir las señales nerviosas del cuerpo y eso se pasa un dispositivo ese dispositivo finalmente a un procesador que es el que procesa finalmente estos datos O sea que estamos hablando como de accesorios si se quiere Esa es la biotecnología como más allá del Software Okay no está centrado en programación código o herramientas no code de Inteligencia artificial sino que es algo más tangible OK Bueno y la biomedicina que son las aplicaciones al diagnóstico no en este caso el diagnóstico de la persona tetrapléjica vaya a saber cuál era el diagnóstico Pero bueno digamos parálisis de piernas medicina de precisión Sí esta es otra de la de las y ya vamos terminando si ahora les voy a comentar Data muy interesante sobre donde estudiar esto pero este es el último ejemplo que les traigo medicina de precisión básicamente incluso acá en Argentina se está empezando a implementar esto es el seguimiento personalizado de la historia clínica de cada persona Recuerden que los diagnósticos que se hacen hoy en día en medicina estamos hablando de muestreo de una cierta cierto grupo de personas y a la mayoría que tiene ciertos síntomas se le asigna un diagnóstico pero no siempre se cumple Ok entonces acá estamos hablando de no generalizar los diagnósticos sino medicina de precisión de precisar o personalizar el diagnóstico a cada paciente porque no todos reaccionan igual a los medicamentos no todos tienen los mismos síntomas frente a la misma enfermedad Ok entonces básicamente la medicina de precisión se trata de un enfoque emergente para el tratamiento de prevención de enfermedades que toma en cuenta la variabilidad individual de los genes el ambiente y el estilo de vida de cada persona Bien voy a tomar un poquito de agua y vamos a hablar rapidito sobre qué Necesito aprender bien estamos para los que vienen del mundo de la informática y les interesa este tema qué necesitan aprender bueno en pantalla vemos y para los que nos están viendo la pantalla les voy a leer Necesito aprender un poco sobre conceptos básicos Ok porque eso después a medida que ustedes van estudiando y capacitándose van a ir incorporándolos no pero necesitamos saber un poquito sobre biología general de psicoquímica estadística probabilidad que les comentaba hoy O sea no solo para las ciencias biológicas para el mayor learning y también no para el mundo de datos en general bases de genéticas biología celular biología molecular teorías de la evolución es un tema muy interesante hablar también de cómo evoluciona con el a medida que pasa el tiempo cada una de las especies no voy a entrar mucho en detalle porque me encanta hablar de estos temas química orgánica y bioquímica básica Estos son algunos de los temas que Estaría bueno como ir incorporando si vienen del mundo de la biología medicina nutrición etcétera y no tiene ni idea de informática qué necesitan aprender sistemas operativos de basados en unix sí básicamente todo lo que no sea de Microsoft Se podría decir ahora Microsoft se incorporó sistemas operativos para poder utilizar Mejor dicho el sistema operativo Linux en su en su en su en su sistema operativo en general pero se utiliza mucho Linux también Mac no pero se utiliza muchísimo Linux en este mundito así que está bueno aprender sobre sobre este sistema operativo por lo tanto líneas de comandos un lenguaje de programación puede ser python r que son los más utilizadas No necesariamente solo eso pero son las más utilizadas y donde hay más herramientas para trabajar en datos biológicos en especial r la luz estadísticos y a los biólogos le encanta r yo en particular me gusta más python pero bueno para gustos colores Júpiter notebooks y manejar notebooks en general puede ser de Júpiter puede ser de cualquier otro otro tipo de software pero notebooks git have o git lapse otras herramientas de control de agresiones expresiones regulares hasta yo los odio la verdad es muy muy complicado de estudiarlos pero está bueno saber expresiones regulares Okay archivos de texto que es donde se guarda la información genética ojo archivo fasta.bet.pdb son algunos de los archivos con los que se trabaja bueno obviamente Inteligencia artificial y está buenísimo es lo ideal saber un poquito de cálculo de álgebra lineal una de las herramientas favoritas a mí me encanta lo estuve estudiando hace No mucho tampoco es que soy súper grosa con estas herramientas pero yo que vengo del mundo de python sí la librería Bio python está muy buena estuve analizando el código genético del coronavirus Sí la verdad que está muy bueno tengo una charla en mi portafolio luego les comparto si quieren me pueden buscar como la talla punto deb ahí en mi página web está o en mis redes sociales Pero bueno estuve grabando una charla para nerdearla que es una conferencia de nerds de acá de Argentina donde estuve aplicando Bio python para decodificar un poquito el coronavirus Así que Les recomiendo que que averigüen qué tal esta herramienta Bio python.org que está muy buena lecturas recomendadas sí vida punto exe de bueno de entre varias de sus autores Y coautores tenemos a Germán González Nicolás andaburu y nicoláspoli Nicolás palos hace poco tuvo el placer de conocerlo En una conferencia de informática es un son personas que son miembros de la del RFC Argentina que es la comunidad que les comenté hace un rato donde soy miembro y otro libro también de dos personas que admiro mucho que son Sebastián bassi Virginia González de python para bioinformática que bueno acá ya es un poquito más técnico no el vida.exe es un libro tal vez más de de más teórico pero a la vez con un lenguaje muy de para personas que no vienen del mundo ni de la informática en las ciencias biológicas sino son conceptos explicados de una manera muy divertida y fácil Clara y python para había informática de vacío González que es un libro por ahí más técnico donde hablan de Bio payton justamente y la verdad que bueno son dos personas que admiro mucho también no tuve el placer de conocerlos en persona pero hablamos mucho en redes y siempre están disponibles para cualquier pregunta que tengan al respecto respecto Bueno hablemos rapidito sobre la situación bioinformática en hispanoamérica Lamentablemente en Argentina ha sido si necesitas mínimo un título de grado y ni hablar de posgrado para ejercer la bioinformática entiendo que es la situación de la mayoría de los países de hispanoamérica al menos bueno latinoamérica incluyendo España no es posible estudiar carreras de grado de posgrado de muchos países hispanoamericanos en todos de momento bueno sabemos que hay ofertas de pregrado sí es decir tecnicaturas eh en chile así que muchas personas de chile acá si les interesa averigüen yo no sé en qué universidad pero sé que las hay pocas carreras de grado incluyen un título intermedio como es en el caso de la Universidad Católica de Córdoba de Argentina o incluso la Universidad Nacional de quilmes que es donde yo estoy inscripta tenemos un título intermedio que es la la tecnicatura envió informática pero para eso también tenés que completar el grado de la licenciatura no para las materias son afines Ok existen ofertas públicas y privadas en todas hispanoamérica las licenciaturas suelen durar entre o de grado No necesariamente licenciaturas capaz algún país tenga alguna ingeniería pero más o menos suelen durar entre tres y cuatro años en algunos cinco como mucho la licenciatura de la Universidad de quilmes dura cinco también en Argentina que es una universidad privada tienen esa opción y también existen diversos cursos y mentorías impulsados por las comunidades sí tienen un montón de comunidades de las que vamos a hablar ahora que son bueno la asociación de Argentina de bioinformática y biología computacional [Música] que es una empresa Pero bueno también tiene una comunidad muy muy ocupada Woman envió informatics de latinoamérica atg genomics que ellos tienen también un podcast muy interesante y donde soy miembro yo el Ace Cívico que es la institución madre que abarca todos los rcg o regional student de de todo el mundo Sí donde estábamos el de Argentina el de chile el de España el de Perú entre otros países hace poco habrían bangladesh en India en muchos lados estaba presente acá si quieren saber un poquito más sobre sobre esta sobre esta sesión o institución Sí y scv b larga.org y ahí tienen toda la información También tienen eventos en Argentina tenemos un hospital que está súper adelantado sobre el tema de herramientas bioinformáticas y y bueno medicina personalizada no que es el hospital italiano Sí donde se hacen la mayoría de las investigaciones Además del conicet y hace un evento anual que es el j y ssumith en este caso tenemos el de 2023 que se va a hacer en noviembre son tres días si no me equivoco que bueno Les recomiendo también hospital italiano.org y ahí tienen toda la información al respecto Muchísimas gracias llegamos Justo a los 40 minutos si tienen alguna pregunta al respecto tengo un ratito para responderlas Muchas gracias Natalia Ay sí pueden hacer clic en la manito Y seguramente consultar o alguna duda que tengan algo que quieran volver a revisar o a ver está abierto el espacio inclusive si quieren dar algún comentario o algo que les llamó la atención también hacen clic en la manito que es justamente para para hablar y ahí estamos atentas tengan vergüenza chicos bueno y justo en el en los comentarios estaba seba consultando acerca de bueno los cursos sobre ya y su relación con la vía informática que ahí no estuviste nombrando principalmente en este caso no son cursos Pero serían como invitaciones no talleres no Natalia los últimos que nombraste y son eventos Sí el del hospital italiano elegí sumit 2023 es un evento que se hace bueno híbrido hoy en día se hace emitido antes hacía presencial en Buenos Aires Así que si entran al hospital italiano.org van a poder inscribirse entiendo que es gratuito al menos las veces que yo estuve gratuito y pueden ver todas las últimas investigaciones al respecto las comunidades que nombré como el isbc Grove o los rcg de cada país son grupos de estudiantes no solo de grado de pregrado como en el caso sino también de posgrado postdoterado sí gente de todos los niveles que se dedican a esta área de una manera dentro de la investigación no de una manera más y como un nivel es mucho más grandes de la parte de investigación que pueden sumarse y tienen muchas actividades muy interesantes para capacitarse como cursos talleres simposios todo el año pasado estuve participando en la provincia de corrientes en un simposio internacional que estuvo muy bueno para hacer un poquito a networking en Chile me imagino que si tienen carreras de pregrado es el único país que tiene deben de tener también este tipo de eventos consulten rcg chile o el rfg de cada país Así que les invito a sumarse ahí que tienen un montón de cursos muy interesantes pero recuerden que para ejercer por el momento necesitan mínimo una carrera de una carrera de pregrado de grado Ok esa es uno de los contras pero pueden ir estudiando de cursos Mientras tanto bien muchas gracias y ahí Fer consulta si hay alguna experiencia para ingresar a este mercado laboral y porque él por ejemplo viene del lado biológico ya tiene una licenciatura Y actualmente está avanzado en una diplomatura en ciencia de datos en la tiene todas las herramientas me encanta muy buena universidad esa la universidad de San Martín si no me equivoco de Argentina Sí yo tiene todas las herramientas para para entrar a la vía informática que le escriba a los chicos del rcg Argentina y o con estas personas que les comenté por redes Virginia González Sebastián y Nicolás paleópoli Germán González son personas que le pueden ayudar ya o que mandan algún correo al conicet todo el tiempo están compartiendo búsquedas dentro de lo privado dentro de Argentina también tiene tienen a stam que hace investigaciones muy interesantes al respecto Pero si ya ven con una base de nuevo de grado en lo posible de grado si bien también de pregrado como lo comenté si ya venís con una base Universitaria es cuestión de sea de informática de ciencias biológicas ya tenés todo lo necesario ahora si ya hiciste cursos o ya entras con algún curso base tenés obviamente un currículum mucho más interesante donde probablemente gane los concursos no porque todo esto se hace por concursos normalmente al menos la parte pública eligen los mejores currículums y ahí entran a proyectos de investigación no hay mercado laboral dentro de lo privado dentro de fuera de lo que sea investigación Okay Por el momento buenísimo y también otra pregunta para volver a escuchar la La charla va a estar grabada la vamos a compartir y otra que es si hay un emprendimientos relacionados a vía informática acá en lata sabes alguno Natalia llegaste a ver hay qué movimiento a nivel de emprendimiento hay dentro del campo no no hay muchos dentro bien formal les comentaba esta empresa que creo que se escribe con que tienen investigaciones Y algún que otro producto pero sé que son productos propios porque hacen investigaciones internas y no no están publicadas Ok yo estuve haciendo un proceso de entrevistas con ello lamentablemente no no vivo en Buenos Aires Así que no pude sumarme al Team pero me estuvieron comentando que sí tienen algún que otro producto pero se enfocan en la investigación y no hay emprendimientos Ok más que eso hoy en día a ver la carrera había informática es muy reciente y se enfoca solo en investigación por el momento sí Habría que ver ahí la verdad que no estoy al tanto si Benja ofenda quieren comentar algo si si llegaron a ver algún emprendimiento relacionado a vía informática están también abierto el escenario para que Comenten hacen la Data los que sepan que me interesa también los siguientes comentarios la verdad es que lograste Sí tal cual lograste responder justo también lo que tenían un formulario Respecto a los libros a Cómo empezar a formarse dentro del campo que eso también va a estar disponible para compartirlo y había una pregunta que hace el análisis de datos también les cuento un poquito rapidito todos los roles de datos sí ingeniero en datos Data scientist análisis de datos con todas áreas si ustedes vienen de esas áreas o les interesa está buenísimo que lo estudien fue el camino que yo elegí mientras me formo en la carrera porque te da muchas herramientas como les había comentado anteriormente el Linux un lenguaje de programación para visualización también se usa mucho si bien se usa más r también son herramientas que les puede llegar a servir eso más que nada dentro del análisis de datos para para empezar en este mundito desde el lado del avión informática porque Recuerden que tienen dos caminos de lo Bio o de la info pero son esos son herramientas que las pueden estudiar en cualquier curso de internet en YouTube o Inteligencia artificial por favor métanse al curso de My Future allá y que está muy bueno tienen otras maneras de estudiarlo Más allá de la facultad no genial ahí justo estaban respondiendo algo que había preguntado Nico que es Cómo empezar en el mundo de la bioinformático qué curso básico como ya es la alumna de pregrado de biología alumnos productos que justo esto de ir incorporando estas herramientas bien Además de los de los libros o de las comunidades que les había comentado antes en cursera tienen entiendo que no es gratuito con todas las personas tienen crucera yo al menos sé que podés cursarlo gratis pero para el certificado que en el mundo del avión informática sí es importante los certificados Obviamente que tenés que pagar para acceder No pero en crucera tenés un par A ver déjame un segundo que los Estoy buscando porque tengo en Twitter un montón de información al respecto en segundo precursores tienen algunos algunos cursos no hay muchos online no hay muchos así que son dos o tres que están en esa web creo con la desde la universidad Harvard también pero no más que eso por ahí comentando también nos comentan que por ejemplo en Guatemala hay un posgrado en bien informática por la universidad de San Carlos Estaría bueno saber si es totalmente remoto si es necesario estar o híbrido o virtual Cuál es la modalidad de esa del posgrado y en Ecuador y maestría en biología computacional muy interesante y es online muy muy buen dato se va muy bueno acá tengo el nombre del curso de cursomática de la Universidad de San Diego para que lo busquen yo lo yo lo hice por la mitad de ese está muy bueno muy interesante para para empezar de cero la verdad Bueno las competiciones de cada eso es otro otro comentario que Les recomiendo también genial Bueno ahí está ginet Esperamos que nos está comentando Ahí está más datos de dónde empezar a estudiar por ejemplo está la maestría informática en la Universidad Nacional en Colombia y también hay una maestría en biología computacional también en la universidad de los Andes también desde Colombia excelente montón de edad me encanta porque yo tengo alguna que otra Data de Argentina pero está bueno que aporten un poquito más sobre la realidad de su país Sí ahí está también Inclusive la oportunidad de estudiarlo online que está como comentaba seba de la Universidad Católica de Ecuador se me estaba olvidando también hace poco hizo un curso de la Universidad de valencia creo que era de valencia que era online era un mooc Ok era un curso corto que no lo abre todo el tiempo ahora está cerrado pero lo había hecho de marzo hasta mayo cada tanto suero en abrir también un curso online y gratuito creo que para el certificado hay que pagar pero para empezar en el mundo también está bueno no como un curso bastante introductorio tal cual y podemos cerrar con esta pregunta porque hace seba que es Cuál es la diferencia entre bioinformática y biología computacional que dice que nunca le quedó claro Cuál es la Cuáles son los puntos más comparando entre estas dos ramas de vida informática y biología computacional les voy a matar con esta respuesta para mí son lo mismo cuando hablamos de biología en sí obviamente estamos hablando de la aplicación de de herramientas a la biología computacional y la bioinformática ya bueno de software No pero en esencia son lo mismo La verdad tampoco es que la tengo Claro si es que hay una gran diferencia al respecto si las hay no les voy a mentir No las sé [Música] para seguir investigando porque es un muy buen tema y al principio por ahí está abierto para seguir comentando la verdad que muchas gracias Natalia esta este esta charla va a estar grabada y la vamos a compartir también va a estar disponible si querés compartirnos el material para que puedan seguir profundizando a las personas que puedan seguir buscando información formándose en esta área que también está muy solicitada y que por lo que vemos también es inclusive podemos desarrollar los emprendimientos o ir innovando dentro de las ciencias biológicas y muchas gracias muchas gracias a todos los que están estuvieron escuchando estuvieron en esta hora Natalia sé que ya tenés que dar una clase Así que muchas gracias Y seguramente dentro de dos semanas otra vez vamos a estar haciendo otro evento relacionado análisis de datos Así que estén muy atentos a la comunidad que lo vamos a estar compartiendo y Esto fue todo gracias estamos hablando Gracias a todos Ahora sí genial nos vemos', metadata={'source': 'pfjWK5ojbRE', 'title': 'Hablemos de IA en bioinformática', 'description': 'Unknown', 'view_count': 21, 'thumbnail_url': 'https://i.ytimg.com/vi/pfjWK5ojbRE/hq720.jpg?v=64bb257c', 'publish_date': '2023-07-21 00:00:00', 'length': 1479, 'author': 'Myfuture-AI'})]




```python
loader_note.load()
```

## 6. Evaluation and LangSmith

- Caso: Bot para servicio al cliente
- ¿Qué prompt me sirve para mi tienda
- Existen múltiples tipos de evaluación pero en este caso ocuparemos más LLM y prompts para evaluar el resultado del mismo modelo.


```python
from langchain.evaluation import load_evaluator
from langchain.evaluation import EvaluatorType
from langchain.evaluation import Criteria
```


```python
list(Criteria)
```




    [<Criteria.CONCISENESS: 'conciseness'>,
     <Criteria.RELEVANCE: 'relevance'>,
     <Criteria.CORRECTNESS: 'correctness'>,
     <Criteria.COHERENCE: 'coherence'>,
     <Criteria.HARMFULNESS: 'harmfulness'>,
     <Criteria.MALICIOUSNESS: 'maliciousness'>,
     <Criteria.HELPFULNESS: 'helpfulness'>,
     <Criteria.CONTROVERSIALITY: 'controversiality'>,
     <Criteria.MISOGYNY: 'misogyny'>,
     <Criteria.CRIMINALITY: 'criminality'>,
     <Criteria.INSENSITIVITY: 'insensitivity'>,
     <Criteria.DEPTH: 'depth'>,
     <Criteria.CREATIVITY: 'creativity'>,
     <Criteria.DETAIL: 'detail'>]




```python
evaluator_concise = load_evaluator("criteria", criteria="conciseness")
evaluator_concise
```




    CriteriaEvalChain(memory=None, callbacks=None, callback_manager=None, verbose=False, tags=None, metadata=None, prompt=PromptTemplate(input_variables=['output', 'input'], output_parser=None, partial_variables={'criteria': 'conciseness: Is the submission concise and to the point?'}, template='You are assessing a submitted answer on a given task or input based on a set of criteria. Here is the data:\n[BEGIN DATA]\n***\n[Input]: {input}\n***\n[Submission]: {output}\n***\n[Criteria]: {criteria}\n***\n[END DATA]\nDoes the submission meet the Criteria? First, write out in a step by step manner your reasoning about each criterion to be sure that your conclusion is correct. Avoid simply stating the correct answers at the outset. Then print only the single character "Y" or "N" (without quotes or punctuation) on its own line corresponding to the correct answer of whether the submission meets all criteria. At the end, repeat just the letter again by itself on a new line.', template_format='f-string', validate_template=True), llm=ChatOpenAI(cache=None, verbose=False, callbacks=None, callback_manager=None, tags=None, metadata=None, client=<class 'openai.api_resources.chat_completion.ChatCompletion'>, model_name='gpt-4', temperature=0.0, model_kwargs={}, openai_api_key='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', openai_api_base='', openai_organization='', openai_proxy='', request_timeout=None, max_retries=6, streaming=False, n=1, max_tokens=None, tiktoken_model_name=None), output_key='results', output_parser=CriteriaResultOutputParser(), return_final_only=True, llm_kwargs={}, criterion_name='conciseness')




```python
print(evaluator_concise.prompt.template)
```

    You are assessing a submitted answer on a given task or input based on a set of criteria. Here is the data:
    [BEGIN DATA]
    ***
    [Input]: {input}
    ***
    [Submission]: {output}
    ***
    [Criteria]: {criteria}
    ***
    [END DATA]
    Does the submission meet the Criteria? First, write out in a step by step manner your reasoning about each criterion to be sure that your conclusion is correct. Avoid simply stating the correct answers at the outset. Then print only the single character "Y" or "N" (without quotes or punctuation) on its own line corresponding to the correct answer of whether the submission meets all criteria. At the end, repeat just the letter again by itself on a new line.



```python
import json
```


```python
eval_result = evaluator_concise.evaluate_strings(
    prediction="What's 2+2? That's an elementary question. The answer you're looking for is that two and two is four.",
    input="What's 2+2?",
)
print(json.dumps(eval_result, indent=2))
```

    {
      "reasoning": "The criterion is conciseness, which means the submission should be brief and to the point. \n\nLooking at the submission, the respondent has added unnecessary information such as \"That's an elementary question\" and \"The answer you're looking for is that\". \n\nThe concise answer to the question \"What's 2+2?\" would simply be \"4\". \n\nTherefore, the submission does not meet the criterion of conciseness.\n\nN",
      "value": "N",
      "score": 0
    }


### Custom criteria


```python
# Un nuevo evaluador para nustro bot, queremos que sea feliz
custom_criterion_1 = {"happy": "Does the output present a warm and happy tone?"}

eval_chain = load_evaluator(
    EvaluatorType.CRITERIA,
    criteria=custom_criterion_1,
)
query = "Cuéntame una historia corta"
prediction = "Un atardecer dorado, abrazados bajo las estrellas, supieron que su amor era eterno."
prediction_happy = "Las lágrimas cayeron en silencio mientras las promesas se rompían en sus miradas."
```


```python
eval_result = eval_chain.evaluate_strings(prediction=prediction, input=query)
print(json.dumps(eval_result, indent=2))
```

    {
      "reasoning": "The criterion is to assess whether the output presents a warm and happy tone. \n\nThe submission is a short story about two people realizing their love is eternal under a golden sunset and stars. This scenario is generally associated with warmth and happiness. The words \"atardecer dorado\" (golden sunset), \"abrazados\" (embraced), \"estrellas\" (stars), and \"amor eterno\" (eternal love) all contribute to a warm and happy tone. \n\nTherefore, the submission meets the criterion.\n\nY",
      "value": "Y",
      "score": 1
    }



```python
eval_result = eval_chain.evaluate_strings(prediction=prediction_happy, input=query)
print(json.dumps(eval_result, indent=2))
```

    {
      "reasoning": "The criterion is asking if the output presents a warm and happy tone. \n\nThe submission is a short sentence that talks about tears falling and promises being broken. \n\nThis suggests a sad or melancholic tone, not a warm and happy one. \n\nTherefore, the submission does not meet the criterion. \n\nN",
      "value": "N",
      "score": 0
    }



```python
from langchain.smith import RunEvalConfig, run_on_dataset
from langsmith import Client
```


```python
prompt1 = """Eres un asistente comercial llamado HappyFeet para una tienda de pantalones, somos un negocio jovial y alegre.
Nos esforzamos por dar una respuesta calmada y útil para nuestros clientes, por esto mismo, somos destacados en el sector de servicio al cliente.
Información tienda:
- Pedidos solo nacionales, contacta a email@example.com para ver posibilidades de envío internacional.
- Descuentos días martes y jueves, liquidaciones de 15% solo en pantalones, desde 3000-9000 pesos.
Recuerda que nos representas y debemos ayudar siempre, al saludar siempre menciona tu nombre!

Pregunta:
{input}
"""

prompt2 = """Eres un asistente comercial útil y directo, no perdemos tiempo con preguntas que no conocemos, simplemente guíalos a que llamen al número 12345678.
Vendemos productos online, si necesitan saber más pueden perfectamente navegar y no hacerte perder el tiempo.

Pregunta:
{input}
"""
```


```python
def create_chain():
    llm = ChatOpenAI(temperature=0)
    return LLMChain.from_string(llm, 
                                prompt1)

def create_chain_2():
    llm = ChatOpenAI(temperature=0)
    return LLMChain.from_string(llm, 
                                prompt2)
```


```python
example_inputs = [
  "Hola",  
  "Cuánto cuestan los pantalones?",
  "Que día tienes descuentos?",
  "Tienen envío fuera del país?",
  "Hace 2 semanas que esty esperando mi pedido!",
]

client = Client()
dataset_name = "Bot Servicio Cliente Clase Langchain"

# Storing inputs in a dataset lets us
# run chains and LLMs over a shared set of examples.
dataset = client.create_dataset(
    dataset_name=dataset_name, description="Customer service bot prompts.",
)
for input_prompt in example_inputs:
    # Each example must be unique and have inputs defined.
    # Outputs are optional
    client.create_example(
        inputs={"question": input_prompt},
        outputs=None,
        dataset_id=dataset.id,
    )
```


```python
eval_config = RunEvalConfig(
    evaluators=[
        # You can define an arbitrary criterion as a key: value pair in the criteria dict
        RunEvalConfig.Criteria({"happy": "Does the output present a warm and happy tone?"}),
        # We provide some simple default criteria like "conciseness" you can use as well
        RunEvalConfig.Criteria("conciseness"),
        RunEvalConfig.Criteria("helpfulness")
    ]
)
```


```python
run_on_dataset(
    client=client,
    dataset_name=dataset_name,
    llm_or_chain_factory=create_chain,
    evaluation=eval_config,
    verbose=True,
    project_name="llmchain-test-1",
)

run_on_dataset(
    client=client,
    dataset_name=dataset_name,
    llm_or_chain_factory=create_chain_2,
    evaluation=eval_config,
    verbose=True,
    project_name="llmchain-test-2",
)
```

    View the evaluation results for project 'llmchain-test-1' at:
    https://smith.langchain.com/o/c3b17e15-db25-4033-973d-ce8b3f766e93/projects/p/f8a39249-0a9a-47c2-baf6-a6d4eda8c9fe
    [------------------------------------------------->] 5/5
     Eval quantiles:
                 0.25  0.5  0.75  mean  mode
    helpfulness   1.0  1.0   1.0   1.0   1.0
    conciseness   0.0  0.0   0.0   0.0   0.0
    happy         1.0  1.0   1.0   1.0   1.0
    View the evaluation results for project 'llmchain-test-2' at:
    https://smith.langchain.com/o/c3b17e15-db25-4033-973d-ce8b3f766e93/projects/p/d55963c3-984c-421a-99f8-aad25144601c
    [------------------------------------------------->] 5/5
     Eval quantiles:
                 0.25  0.5  0.75  mean  mode
    happy         0.0  0.0   1.0   0.4   0.0
    helpfulness   1.0  1.0   1.0   1.0   1.0
    conciseness   0.0  1.0   1.0   0.6   1.0





    {'project_name': 'llmchain-test-2',
     'results': {'595f2f4d-6492-4685-9855-510c27a3658c': {'output': {'input': 'Hace 2 semanas que esty esperando mi pedido!',
        'text': 'Lamentamos mucho la demora en la entrega de tu pedido. Para poder ayudarte con este tema, te recomendamos que te pongas en contacto con nuestro servicio de atención al cliente llamando al número 12345678. Ellos podrán brindarte información actualizada sobre el estado de tu pedido y resolver cualquier duda que puedas tener. ¡Gracias por tu comprensión!'},
       'input': {'question': 'Hace 2 semanas que esty esperando mi pedido!'},
       'feedback': [EvaluationResult(key='happy', score=0, value='N', comment="The criterion is to assess whether the output presents a warm and happy tone. \n\nThe submission starts with an apology for the delay in delivery, which shows empathy and understanding. It then provides a solution by recommending the user to contact customer service for further assistance. The tone throughout the response is polite and professional, aiming to resolve the user's issue. \n\nHowever, the tone of the response can be considered more neutral or professional rather than warm and happy. The response is courteous and helpful, but it does not necessarily convey a sense of joy or happiness. \n\nTherefore, based on the given criterion, the submission does not fully meet the requirement of presenting a warm and happy tone. \n\nN", correction=None, evaluator_info={'__run': RunInfo(run_id=UUID('a4efd3c1-b18f-43bc-b288-f61542079a59'))}, source_run_id=None),
        EvaluationResult(key='helpfulness', score=1, value='Y', comment="The criterion for this task is helpfulness. The submission should be helpful, insightful, and appropriate.\n\nLooking at the submission, it is a response to a complaint about a delayed order. The response is polite and offers a solution to the problem, which is to contact customer service for more information about the order. This is helpful because it provides the user with a way to get more information and possibly resolve the issue.\n\nThe response is also insightful because it acknowledges the problem and offers a solution. It shows understanding of the user's frustration and offers a way to help.\n\nThe response is appropriate because it addresses the user's complaint directly and offers a solution. It is also polite and professional, which is appropriate for a customer service interaction.\n\nBased on this analysis, the submission meets the criterion of being helpful, insightful, and appropriate.\n\nY", correction=None, evaluator_info={'__run': RunInfo(run_id=UUID('37fea443-906a-44ff-9c99-cace1dcd3435'))}, source_run_id=None),
        EvaluationResult(key='conciseness', score=1, value='Y', comment='The criterion is conciseness, which means the submission should be brief and to the point. \n\nLooking at the submission, it starts with an apology for the delay, which is relevant to the input. It then suggests a solution, which is to contact customer service for more information. This is also relevant and necessary. The submission ends with a thank you for understanding, which is a polite way to end the message. \n\nWhile the submission is a bit lengthy, every part of it is necessary and relevant to the input. There is no unnecessary information or filler text. \n\nTherefore, the submission can be considered concise, as it provides all the necessary information in a clear and direct manner. \n\nSo, the submission meets the criterion of conciseness. \n\nY', correction=None, evaluator_info={'__run': RunInfo(run_id=UUID('1d8ed578-ffc8-4736-ac4f-f393f1d3d686'))}, source_run_id=None)]},
      'e99062cd-d95e-49d4-95fd-bc6f765e72e3': {'output': {'input': 'Tienen envío fuera del país?',
        'text': 'Sí, tenemos envío fuera del país. Para obtener más información sobre los detalles y costos de envío internacionales, te recomendamos visitar nuestro sitio web o llamar al número 12345678.'},
       'input': {'question': 'Tienen envío fuera del país?'},
       'feedback': [EvaluationResult(key='happy', score=0, value='N', comment='The criterion is to assess whether the output presents a warm and happy tone. The submission is a response to a question about international shipping. The response is polite and informative, providing the necessary details and guiding the user to find more information. However, it does not necessarily convey a warm and happy tone. It is more neutral and professional. Therefore, the submission does not meet the criterion.\n\nN', correction=None, evaluator_info={'__run': RunInfo(run_id=UUID('eb9f07d6-cba1-4269-9e6d-885822f4bedd'))}, source_run_id=None),
        EvaluationResult(key='helpfulness', score=1, value='Y', comment="The criterion for this task is helpfulness. The submission should be helpful, insightful, and appropriate.\n\nLooking at the submission, it is a response to a question asking if they ship outside the country. The response confirms that they do ship internationally and provides additional information on how to get more details about the international shipping costs and procedures. This is helpful as it answers the question directly and provides further guidance.\n\nThe response is also insightful as it directs the inquirer to the website or a phone number for more detailed information. This shows that the responder understands that the inquirer might need more specific information that can't be fully covered in a brief response.\n\nLastly, the response is appropriate. It directly addresses the question and provides relevant information in a polite and professional manner.\n\nTherefore, the submission meets the criterion of being helpful, insightful, and appropriate.\n\nY", correction=None, evaluator_info={'__run': RunInfo(run_id=UUID('887eab95-41f3-4edc-b09f-f75290048d56'))}, source_run_id=None),
        EvaluationResult(key='conciseness', score=1, value='Y', comment='The criterion is conciseness. The submission should be concise and to the point. \n\nLooking at the submission, it does answer the question directly at the beginning by saying "Sí, tenemos envío fuera del país." This is a concise answer to the question. \n\nHowever, the submission goes on to provide additional information about where to find more details and costs of international shipping. This additional information, while potentially useful, is not strictly necessary to answer the question and therefore could be seen as not being concise.\n\nOn the other hand, this additional information could be seen as being helpful and providing a more complete answer to the question, even if it is not the most concise answer.\n\nTherefore, the assessment of whether the submission meets the criterion of conciseness could depend on how strictly the criterion is interpreted. If a strict interpretation is used, then the submission may not meet the criterion. If a more lenient interpretation is used, then the submission may meet the criterion.\n\nY', correction=None, evaluator_info={'__run': RunInfo(run_id=UUID('16276444-f54c-4e32-b48d-08921fb05316'))}, source_run_id=None)]},
      'ebc8e055-2762-4627-b3d0-98342ae4a9d0': {'output': {'input': 'Que día tienes descuentos?',
        'text': 'Lamentablemente, no puedo responder a esa pregunta ya que soy un asistente de inteligencia artificial y no tengo acceso a información en tiempo real sobre los descuentos. Te recomendaría visitar nuestro sitio web o contactar a nuestro equipo de atención al cliente al número 12345678 para obtener información actualizada sobre los descuentos disponibles.'},
       'input': {'question': 'Que día tienes descuentos?'},
       'feedback': [EvaluationResult(key='happy', score=0, value='N', comment='The criterion is to assess whether the output presents a warm and happy tone. \n\nLooking at the submission, the AI assistant politely explains that it cannot provide real-time discount information. It then suggests visiting the website or contacting customer service for updated information. \n\nWhile the response is polite and helpful, it does not necessarily convey a "happy" tone. The tone is more neutral and informative. \n\nTherefore, based on the given criterion, the submission does not meet the criteria. \n\nN', correction=None, evaluator_info={'__run': RunInfo(run_id=UUID('7e58fdfe-ccf3-4c7c-b5fb-8fdbaa404396'))}, source_run_id=None),
        EvaluationResult(key='helpfulness', score=1, value='Y', comment='The criterion for this task is helpfulness. The submission should be helpful, insightful, and appropriate.\n\nLooking at the submission, the AI assistant clearly states that it cannot provide real-time information on discounts as it does not have access to such data. This is an appropriate response as it is honest and clear about the limitations of the AI assistant.\n\nThe AI assistant then provides a helpful suggestion to the user to visit the website or contact the customer service team for updated information on available discounts. This is insightful as it directs the user to where they can find the information they need.\n\nTherefore, the submission meets the criterion of being helpful, insightful, and appropriate.\n\nY', correction=None, evaluator_info={'__run': RunInfo(run_id=UUID('3a3aa98f-9ff1-4d43-b9cb-0ea2816556bf'))}, source_run_id=None),
        EvaluationResult(key='conciseness', score=0, value='N', comment='The criterion is conciseness, which means the submission should be brief and to the point. \n\nLooking at the submission, the AI assistant provides a detailed explanation as to why it cannot provide the information asked for. It then suggests an alternative way for the user to get the information they need. \n\nWhile the response is detailed, it is not necessarily concise. It provides more information than what was asked for in the input. The user simply asked when discounts are available, and a more concise response could have been: "I\'m sorry, but as an AI, I don\'t have access to real-time discount information. Please check our website or contact customer service."\n\nTherefore, the submission does not meet the criterion of conciseness.\n\nN', correction=None, evaluator_info={'__run': RunInfo(run_id=UUID('6e24fc87-3e7a-4c1a-b4ea-f13cc6f7ae35'))}, source_run_id=None)]},
      'ed48d8ef-f213-42c1-a27a-5aed1d812d30': {'output': {'input': 'Cuánto cuestan los pantalones?',
        'text': 'Nuestros precios varían dependiendo del modelo y la marca de los pantalones. Te recomendaría visitar nuestra página web para obtener información detallada sobre los precios y las opciones disponibles. Si tienes alguna otra pregunta o necesitas ayuda adicional, no dudes en llamarnos al número 12345678. Estaremos encantados de asistirte.'},
       'input': {'question': 'Cuánto cuestan los pantalones?'},
       'feedback': [EvaluationResult(key='helpfulness', score=1, value='Y', comment='The criterion for this task is helpfulness. The submission should be helpful, insightful, and appropriate.\n\nLooking at the submission, the response is helpful as it provides information on where to find the prices of the pants. It also offers additional assistance if needed, which is insightful. The response is appropriate as it answers the question asked in a polite and professional manner.\n\nTherefore, the submission meets the criterion.\n\nY', correction=None, evaluator_info={'__run': RunInfo(run_id=UUID('049f6dbe-aaef-40bd-b6d6-da057f263528'))}, source_run_id=None),
        EvaluationResult(key='conciseness', score=0, value='N', comment='The criterion to be assessed is conciseness, which refers to the submission being concise and to the point. \n\nLooking at the submission, it does answer the question asked, which is about the cost of the pants. However, the response is not concise. It includes additional information about visiting the website for detailed information, offering further assistance, and providing a contact number. While this information might be helpful, it is not directly related to the question asked and therefore makes the response less concise. \n\nTherefore, the submission does not meet the criterion of conciseness. \n\nN', correction=None, evaluator_info={'__run': RunInfo(run_id=UUID('fcfc933b-7743-432c-86be-a7176eda3eba'))}, source_run_id=None),
        EvaluationResult(key='happy', score=1, value='Y', comment='The criterion is to assess whether the output presents a warm and happy tone. \n\nLooking at the submission, the response is polite and helpful. The use of phrases like "Te recomendaría visitar nuestra página web", "Si tienes alguna otra pregunta o necesitas ayuda adicional, no dudes en llamarnos" and "Estaremos encantados de asistirte" convey a friendly and welcoming tone. \n\nThe tone can be considered warm as it is inviting the customer to seek further assistance and is expressing eagerness to help. \n\nHowever, the tone does not necessarily convey happiness. It is professional and courteous, but not explicitly joyful or cheerful. \n\nTherefore, while the response is warm, it may not fully meet the criteria of being happy.\n\nY', correction=None, evaluator_info={'__run': RunInfo(run_id=UUID('3ffd9aa6-54d8-436a-bfb5-40d8b2feb61f'))}, source_run_id=None)]},
      '9b010650-4b85-4439-811f-33c770d42852': {'output': {'input': 'Hola',
        'text': '¡Hola! ¿En qué puedo ayudarte hoy?'},
       'input': {'question': 'Hola'},
       'feedback': [EvaluationResult(key='happy', score=1, value='Y', comment='The criterion is to assess whether the output presents a warm and happy tone. \n\nThe submission is "¡Hola! ¿En qué puedo ayudarte hoy?" which translates to "Hello! How can I help you today?" in English. \n\nThe greeting "¡Hola!" is a standard, friendly greeting in Spanish. \n\nThe question "¿En qué puedo ayudarte hoy?" is a polite and helpful phrase, offering assistance to the person being addressed. \n\nBoth the greeting and the question contribute to a warm and happy tone, as they are friendly, polite, and helpful. \n\nTherefore, the submission meets the criterion. \n\nY', correction=None, evaluator_info={'__run': RunInfo(run_id=UUID('cca10284-9166-4758-aa26-f4266f969044'))}, source_run_id=None),
        EvaluationResult(key='conciseness', score=1, value='Y', comment='The criterion is conciseness, which means the submission should be brief and to the point. \n\nThe input is "Hola", which is Spanish for "Hello". \n\nThe submission is "¡Hola! ¿En qué puedo ayudarte hoy?", which translates to "Hello! How can I help you today?" in English. \n\nWhile the submission does include the input, it also adds an additional question, making it longer than the input. \n\nHowever, the added question is a common follow-up to a greeting and could be seen as part of being to the point in a conversation. \n\nTherefore, the submission could be seen as concise and to the point in the context of a conversation, even though it is longer than the input. \n\nY', correction=None, evaluator_info={'__run': RunInfo(run_id=UUID('bba7020a-2366-4184-9633-7cba60426901'))}, source_run_id=None),
        EvaluationResult(key='helpfulness', score=1, value='Y', comment='The criterion for this task is "helpfulness". The submission should be helpful, insightful, and appropriate.\n\nLooking at the input, it\'s a simple greeting in Spanish: "Hola". \n\nThe submission in response to this input is: "¡Hola! ¿En qué puedo ayudarte hoy?" This translates to "Hello! How can I help you today?" in English.\n\nThis response is appropriate as it is a polite and standard reply to a greeting. It also offers help, which makes it helpful. \n\nThe response might not be particularly insightful, as it doesn\'t provide any deep or unique information. However, given the simplicity of the input, there\'s not much room for insight. The response is as insightful as it can be in this context.\n\nTherefore, the submission meets the criterion of being helpful, insightful, and appropriate.\n\nY', correction=None, evaluator_info={'__run': RunInfo(run_id=UUID('f03ee95e-d119-472a-b956-154c7ef0f232'))}, source_run_id=None)]}}}



## 7. FastAPI Streaming

- Streaming!


```python

```
