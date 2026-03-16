# ============================================================
# A2C (Advantage Actor-Critic) Reinforcement Learning Example
# Environment : CartPole-v1
# Library : Gymnasium + PyTorch + Matplotlib
# ============================================================
# 이 코드는 강화학습 알고리즘 중 하나인 A2C를
# CartPole-v1 환경에 적용한 과제용 최종 통합 예제입니다.
#
# [포함 기능]
# 1. A2C 알고리즘 학습
# 2. 에피소드별 reward 기록
# 3. reward 그래프 저장 및 화면 출력
# 4. 학습된 모델 저장(.pth)
# 5. 저장된 모델 테스트
# 6. render_mode="human" 으로 실제 동작 확인
#
# [A2C 핵심 개념]
# - Actor : 현재 상태에서 어떤 행동을 할지 확률적으로 결정하는 정책 네트워크
# - Critic : 현재 상태가 얼마나 좋은 상태인지 평가하는 가치 네트워크
# - Advantage : "이번 행동이 기대보다 얼마나 더 좋았는가?"를 나타내는 값
#
# [CartPole-v1 환경]
# 카트를 좌우로 움직여 막대가 넘어지지 않도록
# 최대한 오래 균형을 유지하는 환경입니다.

import gymnasium as gym
# Gymnasium은 강화학습 환경을 제공하는 라이브러리입니다.
# CartPole-v1 같은 대표적인 강화학습 실험 환경을 쉽게 생성할 수 있습니다.

import torch
# PyTorch의 기본 라이브러리입니다.
# 텐서 연산, 자동미분, 모델 저장/불러오기 등에 사용됩니다.

import torch.nn as nn
# 신경망 레이어를 정의하기 위한 모듈입니다.
# nn.Linear 같은 완전연결층을 만들 때 사용합니다.

import torch.optim as optim
# 최적화 알고리즘(optimizer)을 제공하는 모듈입니다.
# 여기서는 Adam optimizer를 사용합니다.

import torch.nn.functional as F
# ReLU, softmax 등의 활성화 함수를 함수 형태로 사용할 수 있게 해줍니다.

from torch.distributions import Categorical
# 정책 네트워크가 출력한 행동 확률을 바탕으로
# 실제 행동을 샘플링하기 위한 확률분포 객체입니다.

import matplotlib.pyplot as plt
# 학습 결과(reward 변화)를 그래프로 시각화하기 위한 라이브러리입니다.


learning_rate = 0.0002
# 학습률입니다.
# 모델 파라미터를 얼마나 크게 업데이트할지 결정합니다.

gamma = 0.99
# 할인율(discount factor)입니다.
# 미래 보상을 현재 가치에 얼마나 반영할지를 결정합니다.

max_episodes = 1000
# 총 학습 에피소드 수입니다.
# 에피소드란 reset부터 종료까지의 한 번의 게임을 의미합니다.

print_interval = 20
# 몇 에피소드마다 평균 score를 출력할지 결정합니다.


# ============================================================
# Actor-Critic Network
# ============================================================
# 하나의 신경망 안에 Actor와 Critic을 함께 구현한 구조입니다.
#
# 입력: 상태(state)
# 출력:
# - Actor -> 각 행동의 확률
# - Critic -> 현재 상태의 가치 V(s)
#
# 이 예제에서는 공유 레이어(fc1)를 먼저 통과한 뒤,
# 그 결과를 Actor용 출력층(fc_pi)와 Critic용 출력층(fc_v)로 보냅니다.

class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        # nn.Module을 상속받는 클래스이므로
        # 부모 클래스 초기화를 먼저 수행합니다.

        # 공유 레이어
        self.fc1 = nn.Linear(4, 128)
        # CartPole의 상태(state)는 총 4개의 값으로 구성됩니다.
        # [카트 위치, 카트 속도, 막대 각도, 막대 각속도]
        #
        # 입력 차원 4 -> 은닉층 차원 128로 변환하는 완전연결층입니다.
        # Actor와 Critic이 공통으로 사용하는 feature extractor 역할을 합니다.

        # Actor (policy)
        self.fc_pi = nn.Linear(128, 2)
        # Actor 출력층입니다.
        # CartPole의 행동(action)은 2개입니다.
        # - 0 : 왼쪽 이동
        # - 1 : 오른쪽 이동
        #
        # 따라서 출력 차원은 2입니다.
        # 이후 softmax를 적용하면 각 행동의 확률이 됩니다.

        # Critic (value)
        self.fc_v = nn.Linear(128, 1)
        # Critic 출력층입니다.
        # 현재 상태의 가치 V(s)를 하나의 실수값으로 출력합니다.

    def pi(self, x):
        # Actor 네트워크 부분입니다.
        # 입력 상태 x를 받아 각 행동의 확률을 반환합니다.

        x = F.relu(self.fc1(x))
        # 공유 레이어를 통과시키고 ReLU 활성화 함수를 적용합니다.

        prob = F.softmax(self.fc_pi(x), dim=-1)
        # 행동별 점수(logit)를 만들고 softmax를 적용하여 확률로 변환합니다.
        # 결과 예시: [0.35, 0.65]

        return prob
        # 각 행동의 확률을 반환합니다.

    def v(self, x):
        # Critic 네트워크 부분입니다.
        # 입력 상태 x를 받아 상태 가치 V(s)를 반환합니다.

        x = F.relu(self.fc1(x))
        # Actor와 동일하게 공유 레이어를 통과시켜 feature를 추출합니다.

        v = self.fc_v(x)
        # 현재 상태의 가치 V(s)를 계산합니다.

        return v
        # 상태 가치 V(s)를 반환합니다.


# ============================================================
# Moving Average 계산 함수
# ============================================================
# reward 그래프를 더 보기 좋게 만들기 위해
# 최근 n개 에피소드 평균을 계산합니다.

def moving_average(data, window_size=20):
    averages = []
    # 이동평균 결과를 저장할 리스트입니다.

    for i in range(len(data)):
        # 현재 인덱스 i 기준으로 window_size 만큼 뒤를 포함한 평균을 계산합니다.

        start_idx = max(0, i - window_size + 1)
        # 처음 구간에서는 인덱스가 음수가 되지 않도록 0과 비교합니다.

        window = data[start_idx:i + 1]
        # 최근 window_size 구간의 reward를 잘라옵니다.

        averages.append(sum(window) / len(window))
        # 해당 구간 평균을 저장합니다.

    return averages
    # 전체 이동평균 리스트를 반환합니다.


# ============================================================
# Reward 그래프 저장 함수
# ============================================================
# 에피소드별 reward와 이동평균 reward를 함께 시각화합니다.
# 그래프를 파일로 저장하고 화면에도 출력합니다.

def save_reward_plot(rewards, filename="a2c_reward_plot.png"):
    avg_rewards = moving_average(rewards, window_size=20)
    # 최근 20개 에피소드 기준 이동평균을 계산합니다.

    plt.figure(figsize=(10, 6))
    # 그래프 크기를 설정합니다.

    plt.plot(rewards, label="Episode Reward")
    # 원본 에피소드 reward 곡선입니다.

    plt.plot(avg_rewards, label="Moving Average (20)")
    # 이동평균 reward 곡선입니다.
    # 이 선을 보면 전체 학습 추세를 더 쉽게 파악할 수 있습니다.

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("A2C Training Reward on CartPole-v1")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # 그래프 기본 설정입니다.

    plt.savefig(filename)
    # 현재 폴더에 그래프를 이미지 파일로 저장합니다.

    plt.show()
    # 그래프를 화면에 출력합니다.


# ============================================================
# A2C Training
# ============================================================
# 실제로 환경과 상호작용하며 학습하는 함수입니다.
#
# 핵심 흐름:
# 1. 현재 상태 s를 입력받아 정책 확률 pi(a|s)를 계산
# 2. 그 확률분포에서 행동 a를 샘플링
# 3. 환경에 행동을 적용해 다음 상태 s'와 보상 r 획득
# 4. Critic으로 V(s), V(s') 계산
# 5. TD Target = r + gamma * V(s') 계산
# 6. Advantage = TD Target - V(s) 계산
# 7. Actor loss와 Critic loss를 더해 학습
# 8. 에피소드 reward 저장
# 9. 학습 종료 후 모델 저장 및 reward 그래프 저장

def train():
    env = gym.make("CartPole-v1")
    # CartPole-v1 환경을 생성합니다.

    model = ActorCritic()
    # Actor-Critic 모델 객체를 생성합니다.

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # Adam optimizer를 생성합니다.

    episode_rewards = []
    # 각 에피소드의 총 reward를 저장할 리스트입니다.
    # 이후 그래프를 그릴 때 사용됩니다.

    score = 0.0
    # 최근 print_interval 구간의 누적 reward를 저장하는 변수입니다.

    for n_epi in range(max_episodes):
        # 총 max_episodes만큼 학습을 반복합니다.

        s, _ = env.reset()
        # 환경 초기화 후 초기 상태를 받습니다.

        done = False
        # 에피소드 종료 여부를 나타내는 변수입니다.

        episode_reward = 0.0
        # 현재 에피소드 하나의 총 reward를 저장합니다.

        while not done:
            # 에피소드가 끝날 때까지 반복합니다.

            s = torch.from_numpy(s).float()
            # NumPy 상태를 PyTorch 텐서로 변환합니다.

            prob = model.pi(s)
            # Actor에 현재 상태를 넣어 행동 확률을 얻습니다.

            m = Categorical(prob)
            # 행동 확률을 기반으로 범주형 확률분포를 만듭니다.

            a = m.sample()
            # 확률분포에서 행동 하나를 샘플링합니다.
            # A2C는 학습 시 확률적으로 행동을 선택합니다.

            s_prime, r, terminated, truncated, _ = env.step(a.item())
            # 선택한 행동을 환경에 적용합니다.
            #
            # 반환값:
            # - s_prime : 다음 상태
            # - r : 즉시 보상
            # - terminated : 환경 논리상 종료 여부
            # - truncated : 시간 제한 등에 의한 종료 여부

            done = terminated or truncated
            # 둘 중 하나라도 True이면 에피소드 종료입니다.

            s_prime = torch.from_numpy(s_prime).float()
            # 다음 상태도 텐서로 변환합니다.

            # 현재 상태 가치 V(s)
            value = model.v(s)
            # Critic이 현재 상태의 가치를 예측합니다.

            # 다음 상태 가치 V(s')
            next_value = model.v(s_prime)
            # Critic이 다음 상태의 가치를 예측합니다.

            # TD Target
            td_target = r + gamma * next_value * (1 - done)
            # TD Target을 계산합니다.
            #
            # 기본 식:
            # TD Target = r + gamma * V(s')
            #
            # done=True인 종료 상태에서는 다음 상태 가치가 필요 없으므로
            # (1 - done)을 곱해 0이 되도록 처리합니다.

            # Advantage 계산
            advantage = td_target - value
            # Advantage는 "이번 행동 결과가 예상보다 얼마나 좋았는가"를 나타냅니다.

            # Actor loss
            log_prob = torch.log(prob[a] + 1e-8)
            # 선택한 행동의 확률에 로그를 취합니다.
            # +1e-8은 log(0) 방지를 위한 작은 값입니다.

            actor_loss = -log_prob * advantage.detach()
            # Actor는 Advantage가 큰 행동의 확률을 높이는 방향으로 학습합니다.
            # detach()를 사용해 Critic 쪽 gradient가 Actor에 섞이지 않도록 합니다.

            # Critic loss
            critic_loss = advantage.pow(2)
            # Critic은 TD Target과 자신의 예측값 V(s)의 차이를 줄이는 방향으로 학습합니다.
            # advantage = td_target - value 이므로 제곱 오차 형태가 됩니다.

            # 전체 loss
            loss = actor_loss + critic_loss
            # Actor loss와 Critic loss를 합쳐 한 번에 역전파합니다.

            optimizer.zero_grad()
            # 이전 step의 gradient를 초기화합니다.

            loss.backward()
            # 현재 loss 기준으로 gradient를 계산합니다.

            optimizer.step()
            # gradient를 이용해 모델 파라미터를 업데이트합니다.

            s = s_prime.numpy()
            # 다음 반복을 위해 현재 상태를 다음 상태로 갱신합니다.

            score += r
            # print_interval 구간 평균 계산용 누적 reward입니다.

            episode_reward += r
            # 현재 에피소드의 총 reward에 현재 보상을 더합니다.

        episode_rewards.append(episode_reward)
        # 에피소드가 끝나면 총 reward를 리스트에 저장합니다.

        if n_epi % print_interval == 0 and n_epi != 0:
            print(f"episode : {n_epi}, avg score : {score / print_interval:.2f}")
            # 최근 print_interval 동안의 평균 reward를 출력합니다.

            score = 0.0
            # 다음 구간 평균 계산을 위해 score를 초기화합니다.

    env.close()
    # 학습 환경을 종료합니다.

    torch.save(model.state_dict(), "a2c_cartpole.pth")
    # 학습이 끝난 모델의 가중치를 파일로 저장합니다.

    print("\n학습 완료: 모델이 a2c_cartpole.pth 로 저장되었습니다.")
    # 저장 완료 메시지입니다.

    save_reward_plot(episode_rewards, filename="a2c_reward_plot.png")
    # reward 그래프를 저장하고 화면에 출력합니다.

    print("그래프 저장 완료: a2c_reward_plot.png")
    # 그래프 저장 완료 메시지입니다.

    return model, episode_rewards
    # 필요 시 외부에서 활용할 수 있도록 모델과 reward 리스트를 반환합니다.


# ============================================================
# Test Function
# ============================================================
# 저장된 모델을 불러와 실제로 CartPole을 플레이하는 함수입니다.
#
# 학습 단계에서는 확률적으로 행동을 샘플링했지만,
# 테스트 단계에서는 가장 확률이 높은 행동(argmax)을 선택합니다.
# 이렇게 하면 학습된 정책의 성능을 보다 안정적으로 확인할 수 있습니다.

def test(render=True, episodes=3):
    if render:
        env = gym.make("CartPole-v1", render_mode="human")
        # render=True이면 실제 CartPole 창이 화면에 뜹니다.
    else:
        env = gym.make("CartPole-v1")
        # render=False이면 창 없이 테스트만 진행합니다.

    model = ActorCritic()
    # 테스트용 모델 객체를 다시 생성합니다.

    model.load_state_dict(torch.load("a2c_cartpole.pth"))
    # 학습 중 저장한 가중치를 불러옵니다.

    model.eval()
    # 평가 모드로 전환합니다.
    # 드롭아웃, 배치정규화 등이 있을 경우 평가 방식으로 동작하게 됩니다.
    # 이 예제에서는 큰 차이는 없지만 습관적으로 넣는 것이 좋습니다.

    print("\n===== 테스트 시작 =====")

    for ep in range(episodes):
        s, _ = env.reset()
        # 테스트 에피소드 시작 시 환경을 초기화합니다.

        done = False
        # 에피소드 종료 여부입니다.

        total_reward = 0
        # 테스트 에피소드 총 reward입니다.

        while not done:
            s = torch.from_numpy(s).float()
            # 상태를 텐서로 변환합니다.

            with torch.no_grad():
                prob = model.pi(s)
            # 테스트 시에는 gradient 계산이 필요 없으므로 no_grad()를 사용합니다.

            a = torch.argmax(prob).item()
            # 테스트에서는 가장 확률이 높은 행동을 선택합니다.
            # 학습과 달리 확률 샘플링이 아니라 greedy 방식입니다.

            s_prime, r, terminated, truncated, _ = env.step(a)
            # 선택한 행동을 환경에 적용합니다.

            done = terminated or truncated
            # 종료 여부를 판단합니다.

            s = s_prime
            # 다음 반복을 위해 상태를 갱신합니다.

            total_reward += r
            # 테스트 에피소드 reward를 누적합니다.

        print(f"test episode {ep + 1}, reward : {total_reward}")
        # 테스트 결과를 출력합니다.

    env.close()
    # 테스트 환경을 종료합니다.

    print("===== 테스트 종료 =====")


# ============================================================
# Main
# ============================================================
# 현재 파일을 직접 실행하면 아래 순서로 진행됩니다.
#
# 1. A2C 학습 수행
# 2. reward 그래프 저장 및 출력
# 3. 학습된 모델 저장
# 4. 저장된 모델로 테스트 수행
# 5. render 창으로 실제 CartPole 움직임 확인

if __name__ == "__main__":
    train()
    test(render=True, episodes=3)