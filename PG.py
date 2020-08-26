import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import gym
import time

def sample_trajectory(e, batch_size, model):
    ob_batch = []
    reward_batch = []
    action_batch = []
    for i in range(batch_size):
        ob = e.reset()
        current_obs = [ob]
        current_rewards = []
        current_actions = []
        while True:
            action = torch.distributions.categorical.Categorical(logits=model(torch.from_numpy(ob).cuda()), -1).sample().item()
            ob, reward, done, _ = e.step(action)
            current_rewards.append(reward)
            current_actions.append(action)
            if done:
                break
            current_obs.append(ob)
        ob_batch.append(np.array(current_obs))
        reward_batch.append(np.array(current_rewards))
        action_batch.append(np.array(current_actions))
    return ob_batch, reward_batch, action_batch

def get_advantage(rewards, gamma):
    rewards = [gamma ** i * r for i, r in enumerate(rewards)]
    rewards = [np.sum(rewards[i:]) for i in range(len(rewards))]
    return np.array(rewards)

batch_size = 512
e = gym.make('LunarLander-v2')
model = nn.Sequential(nn.Linear(e.observation_space.shape[0], 64), nn.Linear(64, 64), nn.Linear(64, 64), nn.Linear(64, e.action_space.n)).cuda()
opt = torch.optim.RMSprop(model.parameters(), 2.5e-5, momentum=0.9)

for i in range(200):
    ob_batch, reward_batch, action_batch = sample_trajectory(e, batch_size, model)

    ob_batch = np.concatenate(ob_batch)
    advantage_batch = np.concatenate([get_advantage(i, 0.99) for i in reward_batch])
    action_batch = np.concatenate(action_batch)

    opt.zero_grad()
    logits = model(torch.from_numpy(ob_batch).cuda())
    prob = F.softmax(logits, -1)
    log_prob = torch.log(torch.squeeze(prob[range(logits.shape[0]), action_batch]))
    loss = -torch.sum(log_prob * torch.from_numpy(advantage_batch.astype('float32')).cuda()) / batch_size
    loss.backward()
    opt.step()
    print('iter {} / {}:'.format(i, 200))
    print(np.sum(np.concatenate(reward_batch)) / batch_size)
    print(loss.item())

ob = e.reset()
for _ in range(10000):
    e.render()
    action = torch.argmax(model(torch.from_numpy(ob).cuda()), -1).item()
    ob, _, done, _ = e.step(action)
    time.sleep(0.016)
    if done:
        ob = e.reset()