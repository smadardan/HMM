# HMM

This package is an implementation of a hidden markov models in 2 ways:
1. viterbi ([info here](https://en.wikipedia.org/wiki/Viterbi_algorithm))
2. backward - forward ([further reading](https://en.wikipedia.org/wiki/Forward%E2%80%93backward_algorithm))

## usage
when you see an observation and can't tell what is the hidden state. usually you also have some prior information over the hidden states (pi)
  
for example:
1. we would like to infer the weather given observation of another shoes that can be 1. dirty or clean. We assume two possible states for the weather: state 1 = rain, state 2 = no rain = sunny.
2. the observation are words and the hidden states are the tags (verb, noun, etc). 
you can use the found probabilities to create sentences.

## Usage
1. clone the repository
2. insert items in config/input.json. you can use the variables that are already there or change the numbers and the names of the observations and the hidden states as long as you keep the same form of input. (you can insert as many observations and hidden states as you like)
3. you can change the length of the observations sequence or the number of iterations in the conf/config_vars.py file.
4. open the console. change directory to this repository
5. write:

```
python main.py
```
6. in the end we will receive the hidden stated received for each method (sampling, viterbi and backward-forward) in the logs\log_file.log. (an example is already present there)

## License
[MIT](https://choosealicense.com/licenses/mit/)
