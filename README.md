# AI-Assistant
Zach Gunderson and Aren Desai attempt to make AGI.

## Overview
See [the project description docs.](https://docs.google.com/document/d/1LAKMdX9D1TIlcan3xt2AHcYOopl7pGLNz-sFh7Ijrn0/edit)

## Design Paradigm
OOP-approach. Thinking we have one main loop, running permanently and processing functions. A user can use the default functions and add plugins, which are python functions, by specifying them in a configuration file. Designing new functions for the AIA to call should be easy. 

## LLM Communication
We'll use gRPC. 
