# Nanodegree Engenheiro de Machine Learning
## Proposta de projeto final
Cristian Carlos dos Santos

18 de Agosto de 2019

## AWS DeepRacer Challenge


### Histórico do assunto

As described on the AWS Amazon page itself:

"AWS DeepRacer is a reinforcement learning (RL) -enabled autonomous 1 / 18th-scale vehicle with supporting services in the AWS Machine Learning ecosystem. It offers an interactive learning system for users to acquire and refine their skill set in machine learning in general and reinforcement learning in particular You can use the AWS DeepRacer console to train and evaluate deep reinforcement learning models in simulation and then deploy them to an AWS DeepRacer vehicle for autonomous driving You can also join AWS DeepRacer League to race in the online Virtual Circuit or the in-person Summit Circuit. "

This model of competition and RL learning makes us learn about the subject which is a fascinating but very complex field of study. Recently Udacity launched the AWS DeepRacer Scholarship Challenge and as I had not selected my proposal for the final work I decided to apply it to my work.

### Descrição do problema

The challenge is a race. Initially, the goal is to get the cart to the finish by completing 100% of the track. Afterward, the goal is to improve the lap time by applying optimizations to the reward function and adjustment of hyperparameters.

### Conjuntos de dados e entradas

AWS DeepRacer trains models using the Proximal Policy Optimization (PPO) algorithm. According to the AWS DeepRacer course's "Value Functions" (L4: Reinforcement Learning) video class, this algorithm is used because it is efficient, stable and easy to use compared to other algorithms. The Algorithm uses two neural networks during training: Policy Network (Actor-Network) and Value Network (Critic Network).

- Policy Network: Decides what action to take according to the image received in the input.
- Value Network: Estimates the cumulative result we are likely to get, considering the image as an input.

#### Reward
For the construction of the reward function, we have an input of a variable called "params". This variable is a library in the following format:

```
{
    "all_wheels_on_track": Boolean,    # flag to indicate if the vehicle is on the track
    "x": float,                        # vehicle's x-coordinate in meters
    "y": float,                        # vehicle's y-coordinate in meters
    "distance_from_center": float,     # distance in meters from the track center 
    "is_left_of_center": Boolean,      # Flag to indicate if the vehicle is on the left side to the track center or not. 
    "heading": float,                  # vehicle's yaw in degrees
    "progress": float,                 # percentage of track completed
    "steps": int,                      # number steps completed
    "speed": float,                    # vehicle's speed in meters per second (m/s)
    "steering_angle": float,          # vehicle's steering angle in degrees
    "track_width": float,              # width of the track
    "waypoints": [[float, float], … ], # list of [x,y] as milestones along the track center
    "closest_waypoints": [int, int]    # indices of the two nearest waypoints.
}
```

More information is available at:  https://docs.aws.amazon.com/pt_br/deepracer/latest/developerguide/deepracer-reward-function-input.html

#### Hyperparâmetros
The available hyperparameters for optimization are as follows:
- Gradient descent batch size (Tamanho de lote da descida de gradiente)
- Number of epochs (Número de epochs)
- Learning rate (Taxa de aprendizado)
- Entropy
- Discount factor (Fator de desconto)
- Loss type (Tipo de perda)
- Number of experience episodes between each policy-updating iteration (Número de episódios de experiência entre cada iteração de atualização de política)

More information is available at: https://docs.aws.amazon.com/pt_br/deepracer/latest/developerguide/deepracer-console-train-evaluate-models.html#deepracer-iteratively-adjust-hyperparameters

For this project, I will use the local environment provided in Github: https://github.com/crr0004/deepracer. I intend to use my local machine to train the model and after uploading it to the AWS environment to validate it in operation.

### Descrição da solução

To solve the proposed problem I will use as a starting point the examples of a reward function and hyperparameters already informed in the addresses above. For the reward function, I will use a way to maximize expected actions, such as staying on track, gaining speed and completing laps. For hyperparameters, according to video "Intro to Tuning Hyperparameters" (L5: Tuning Your Model - AWS DeepRacer Course): "Figuring out what works best for your model is usually done through trial and error."

### Modelo de referência (benchmark)

As a reference, the “re: Invent 2018” winner completed the lap in 12.68 sec. For this model the following reward function was used:

```
def reward_function(params):
    track_width = params['track_width']
    distance_from_center = params['distance_from_center']
    all_wheels_on_track = params['all_wheels_on_track']
    speed = params['speed']
    SPEED_THRESHOLD = 1.0 

    
    # Calculate 3 markers that are at varying distances away from the center line
    marker_1 = 0.1 * track_width
    marker_2 = 0.25 * track_width
    marker_3 = 0.5 * track_width
    
    # Give higher reward if the car is closer to center line and vice versa
    if distance_from_center <= marker_1:
        reward = 1.0
    elif distance_from_center <= marker_2:
        reward = 0.5
    elif distance_from_center <= marker_3:
        reward = 0.1
    else:
        reward = 1e-3  # likely crashed/ close to off track

    if not all_wheels_on_track:
		# Penalize if the car goes off track
        reward = 1e-3
    elif speed < SPEED_THRESHOLD:
		# Penalize if the car goes too slow
        reward = reward + 0.5
    else:
		# High reward if the car stays on track and goes fast
        reward = reward + 1.0

    return float(reward)
```

The hyperparameters used were as follows:

- Gradient descent batch size: 64
- Entropy: 0.01
- Discount factor 0.666
- Loss type: Huber
- Learning Rate: 0.0003
- Number of experience episodes between each policy-updating iteration: 20
- Number of epochs: 10

From these benchmarks, I want to walk the path between this great benchmark and the default parameters. If possible, further refine the model and test the results.

Source: https://medium.com/vaibhav-malpanis-blog/how-to-win-at-deepracer-league-code-and-model-included-27742b868794

### Métricas de avaliação

For the evaluation metric, the idea is to use lap completion time, iteration rewards, and algorithm evaluation on different clues. To facilitate this analysis I will use the Jupyter notebook available in the "Using Jupyter Notebook for analyzing DeepRacer's logs" article.

Article available in: https://codelikeamother.uk/using-jupyter-notebook-for-analysing-deepracer-s-logs

### Design do projeto

Nesta seção final, sintetize um fluxo de trabalho teórico para obtenção de uma solução para o problema em questão. Discuta detalhadamente quais estratégias você considera utilizar, quais análises de dados podem ser necessárias de antemão e quais algoritmos serão considerados na sua implementação. O fluxo de trabalho e discussão propostos devem estar alinhados com as seções anteriores. Adicionalmente, você poderá incluir pequenas visualizações, pseudocódigo ou diagramas para auxiliar na descrição do design do projeto, mas não é obrigatório. A discussão deve seguir claramente o fluxo de trabalho proposto para o projeto de conclusão.

O fluxo de trabalho inicial é o seguinte:

- Experimentação do ambiente AWS DeepRacer
- Avaliação do impacto de cada parâmetro da função de recompensa.
- Desenvolvimento da função de recompensa
- Comparação da função de recompensa ótima e da função de recompensa desenvolvida por mim
- Avaliação do impacto dos hyperparâmetros para o modelo.
- Ajustes dos parâmetro conforme o desempenho obtido.
- Comparação dos hyperparâmetros ótimos e dos hyperparâmetros definidos por mim
- Revisão final da função de recompensa e hyperparâmetros com base nos logs obtidos.
- Exportação do modelo final e avaliação do mesmo com base no leaderboard. 
-----------

**Antes de enviar sua proposta, pergunte-se. . .**

- A proposta que você escreveu segue uma estrutura bem organizada, similar ao modelo de projeto?
- Todas as seções (em especial, **Descrição da solução** e **Design do projeto**) estão escritas de uma forma clara, concisa e específica? Existe algum termo ou frase ambígua que precise de esclarecimento?
- O público-alvo de seu projeto será capaz de entender sua proposta?
- Você revisou sua proposta de projeto adequadamente, de forma a minimizar a quantidade de erros gramaticais e ortográficos?
- Todos os recursos usados neste projeto foram corretamente citados e referenciados?
