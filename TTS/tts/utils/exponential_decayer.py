import torch


class ExponentialDecayer():

    def __init__(self, initial_theta=0, initial_lambda=0): #10**(-4.8)):
        self.initial_theta = torch.FloatTensor([initial_theta])
        self.initial_lambda = torch.FloatTensor([initial_lambda])
        self.global_step = 0 

    def get_theta(self):
        return self.initial_theta * torch.exp(-self.initial_lambda * self.global_step)
    
    def step(self):
        self.global_step += 1
    
    def reset(self):
        self.global_step = 0


if __name__ == '__main__':
    exponential_decayer = ExponentialDecayer()
    thetas = []
    for _ in range(int(2e5)):
        thetas.append(exponential_decayer.get_theta())
        exponential_decayer.step()
    print(thetas[::10000])
