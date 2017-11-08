# async_deep_reinforce

This fork tries to make A3C more widely usable with all of the games available in open ai's gym for reinforcement learner.


## About the original

An attempt to repdroduce Google Deep Mind's paper "Asynchronous Methods for Deep Reinforcement Learning."

http://arxiv.org/abs/1602.01783

Asynchronous Advantage Actor-Critic (A3C) method for playing "Atari Pong" is implemented with TensorFlow.
Both A3C-FF and A3C-LSTM are implemented.

## How to build

    $pip install -r requirements.txt

## How to run

To train,

    $python a3c.py configs/space_invaders.json

To display the result with game play,

    $python a3c_disp.py configs/space_invaders.json

## Using GPU
To enable gpu, change "USE_GPU" flag in any of the json-configs.

When running with 8 parallel game environemts, speeds of GPU (GTX980Ti) and CPU(Core i7 6700) were like this. (Recorded with LOCAL_T_MAX=20 setting.)

|type | A3C-FF             |A3C-LSTM          |
|-----|--------------------|------------------|
| GPU | 1722 steps per sec |864 steps per sec |
| CPU | 1077 steps per sec |540 steps per sec |


## Result
Score plots of local threads of pong were like these. (with GTX980Ti)

### A3C-LSTM LOCAL_T_MAX = 5

![A3C-LSTM T=5](./docs/graph_t5.png)

### A3C-LSTM LOCAL_T_MAX = 20

![A3C-LSTM T=20](./docs/graph_t20.png)

Scores are not averaged using global network unlike the original paper.

## Requirements
- TensorFlow r1.0
- numpy
- cv2
- matplotlib
-openai

## References

This project uses setting written in muupan's wiki [muuupan/async-rl] (https://github.com/muupan/async-rl/wiki)


## Acknowledgements

- [@aravindsrinivas](https://github.com/aravindsrinivas) for providing information for some of the hyper parameters.

