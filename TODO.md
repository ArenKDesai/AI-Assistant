# Aren's TODO

## Plugins
I want to organize the LLMs in this way:

Main LLM: Receives a user's message. Python program provides it a list of all "Plugins", or LLMs with access to functions. It chooses which Plugins to send the query. This can be $>=0$ Plugins.  

Plugins: LLMs that receive the query from the Main LLM. These have access to a list of functions, and chooses which Python functions to call and what parameters to specify based on the user input. The response is formatted with JSONformer. 

## Steps:
1. Refine the current "receptionist" LLM to select and run python programs with decent accuracy. 
2. Modify code to be copy-able, maybe with generic classes overwritable classes. This is so a directory or python file can be easily copied an edited to make a new plugin. 
3. Set up the Main LLM (maybe with a cooler name) with a list of plugins available. 
4. Set up this Main LLM as a permanent loop with text input with gRPC, preferably in a Docker container. 
5. Set up Telegram messaging so you can message the Main LLM in the Docker container (or server). Other forms of messaging work too, like email. 
