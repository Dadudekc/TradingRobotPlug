import numpy as np
from risk_management import RiskManager

class TradingEnv:
    def __init__(self, data, initial_balance=10000, risk_manager=None):
        self.data = data
        self.current_step = 0
        self.initial_balance = initial_balance
        self.balance = self.initial_balance
        self.shares_held = 0
        self.total_reward = 0
        self.done = False
        self.price = self.data['Close'].iloc[self.current_step]
        self.risk_manager = risk_manager

        if self.risk_manager:
            self.risk_manager.initialize(self.balance)

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.total_reward = 0
        self.done = False
        self.price = self.data['Close'].iloc[self.current_step]

        if self.risk_manager:
            self.risk_manager.initialize(self.balance)

        return self._get_observation()

    def step(self, action):
        self._take_action(action)
        self.current_step += 1
        self.price = self.data['Close'].iloc[self.current_step]
        reward = self._calculate_reward()
        self.total_reward += reward

        if self.risk_manager:
            self.risk_manager.update(self.balance)
            risk_status = self.risk_manager.check_risk(self.balance)
            if risk_status == "STOP_TRADING":
                self.done = True
            elif risk_status == "STOP_LOSS":
                self._take_action([1, 1])  # Sell all shares
                self.done = True
            elif risk_status == "TAKE_PROFIT":
                self._take_action([1, 1])  # Sell all shares
                self.done = True

        done = self.current_step >= len(self.data) - 1 or self.done
        info = {'balance': self.balance, 'price': self.price}
        return self._get_observation(), reward, done, info

    def _take_action(self, action):
        action_type = action[0]
        amount = action[1]

        if action_type == 0:  # Buy
            total_possible = self.balance // self.price
            shares_bought = total_possible * amount
            cost = shares_bought * self.price * (1 + transaction_cost)
            self.balance -= cost
            self.shares_held += shares_bought
        elif action_type == 1:  # Sell
            shares_sold = self.shares_held * amount
            self.balance += shares_sold * self.price * (1 - transaction_cost)
            self.shares_held -= shares_sold

    def _calculate_reward(self):
        current_value = self.shares_held * self.price + self.balance
        reward = current_value - self.total_reward
        return reward

    def _get_observation(self):
        return np.array([self.balance, self.shares_held, self.price])
