"""
Part 1：信道编码实验

学生需要完成 Hamming(7,4) 编码、伴随式计算和单比特纠错译码。
选做内容包括卷积码编码和 Viterbi 硬判决译码。
"""

import numpy as np
from utils import (
    binary_symmetric_channel,
    calculate_ber,
    generate_bits,
    plot_ber_curve,
)

HAMMING_G = np.array([
    [1, 0, 0, 0, 1, 1, 0],
    [0, 1, 0, 0, 1, 0, 1],
    [0, 0, 1, 0, 0, 1, 1],
    [0, 0, 0, 1, 1, 1, 1],
], dtype=int)

HAMMING_H = np.array([
    [1, 1, 0, 1, 1, 0, 0],
    [1, 0, 1, 1, 0, 1, 0],
    [0, 1, 1, 1, 0, 0, 1],
], dtype=int)

SYNDROME_TO_ERROR = {
    tuple(HAMMING_H[:, j]): j
    for j in range(HAMMING_H.shape[1])
}


def hamming74_encode(bits):
    """
    Hamming(7,4) 系统码编码。

    参数:
        bits: 一维 0/1 数组，长度必须是 4 的倍数。

    返回:
        encoded: 一维 0/1 编码比特数组，长度为输入的 7/4 倍。

    要求:
        使用课件中的生成矩阵 G，按 GF(2) 进行矩阵乘法。
    """
    bits = np.asarray(bits, dtype=int)
    if bits.ndim != 1:
        raise ValueError('bits 必须是一维数组')
    if len(bits) % 4 != 0:
        raise ValueError('Hamming(7,4) 要求输入长度为 4 的倍数')
    if not np.all((bits == 0) | (bits == 1)):
        raise ValueError('bits 只能包含 0 或 1')

    # 将 bits reshape 为 (-1, 4)，再与 HAMMING_G 相乘并对 2 取模
    blocks = bits.reshape(-1, 4)
    encoded = (blocks @ HAMMING_G) % 2
    return encoded.flatten()


def hamming74_syndrome(codewords):
    """
    计算 Hamming(7,4) 码字的伴随式。

    参数:
        codewords: 一维或二维 0/1 数组。若为一维，长度必须是 7 的倍数。

    返回:
        syndromes: 形状为 (N, 3) 的伴随式数组。
    """
    codewords = np.asarray(codewords, dtype=int)
    if codewords.ndim == 1:
        if len(codewords) % 7 != 0:
            raise ValueError('码字长度必须是 7 的倍数')
        codewords = codewords.reshape(-1, 7)
    if codewords.shape[1] != 7:
        raise ValueError('每个 Hamming(7,4) 码字长度必须为 7')

    # 计算 s = r H^T mod 2
    syndromes = (codewords @ HAMMING_H.T) % 2
    return syndromes


def hamming74_decode(received):
    """
    Hamming(7,4) 单比特纠错译码。

    参数:
        received: 一维 0/1 接收序列，长度必须是 7 的倍数。

    返回:
        decoded_bits: 纠错后提取出的信息比特序列。

    提示:
        1. 计算每个码字的伴随式。
        2. 若伴随式非零，将其与 H 的各列比较，定位错误比特。
        3. 翻转对应错误位。
        4. 系统码的信息位为前 4 位。
    """
    received = np.asarray(received, dtype=int)
    if received.ndim != 1 or len(received) % 7 != 0:
        raise ValueError('received 必须是一维数组，长度为 7 的倍数')

    # reshape为(-1, 7)，避免直接修改输入
    codewords = received.reshape(-1, 7).copy()
    syndromes = hamming74_syndrome(codewords)
    
    # 对每个非零伴随式进行纠错
    for i, syndrome in enumerate(syndromes):
        if syndrome.any():
            error_position = SYNDROME_TO_ERROR.get(tuple(syndrome))
            if error_position is not None:
                codewords[i, error_position] ^= 1

    # 取前 4 个信息位并 flatten 返回
    return codewords[:, :4].ravel()


def convolutional_encode(bits):
    """
    选做：实现 (2,1,3) 卷积码编码，生成多项式为 g1=111, g2=101。

    默认在末尾添加 2 个 0 作为尾比特，使状态回到全零。
    """
    bits = np.asarray(bits, dtype=int)
    if not np.all((bits == 0) | (bits == 1)):
        raise ValueError('bits 只能包含 0 或 1')

    # 在末尾添加 2 个尾比特使状态回到全零
    bits_with_tail = np.concatenate([bits, np.array([0, 0], dtype=int)])
    
    # 初始状态：2个秘密状态位
    state = 0  # (s1, s0)
    encoded = []
    
    for bit in bits_with_tail:
        # 根据 g1=111 (7) 和 g2=101 (5) 计算两个输出比特
        # g1=111: 输出1 = input XOR state_bit1 XOR state_bit0
        # g2=101: 输出2 = input XOR state_bit0
        out1 = bit ^ ((state >> 1) & 1) ^ (state & 1)  # g1
        out2 = bit ^ (state & 1)  # g2
        encoded.append(out1)
        encoded.append(out2)
        
        # 更新状态: 新状态 = (input, state_bit1)
        state = (bit << 1) | ((state >> 1) & 1)
    
    return np.array(encoded, dtype=int)


def viterbi_decode_hard(received_bits):
    """
    选做：实现 (2,1,3) 卷积码硬判决 Viterbi 译码。
    """
    received_bits = np.asarray(received_bits, dtype=int)
    if len(received_bits) % 2 != 0:
        raise ValueError('卷积码接收序列长度必须是 2 的倍数')

    # 4个可能的状态：(s1, s0)
    num_states = 4
    num_steps = len(received_bits) // 2
    
    # 处理接收序列，每次输入一寸(2比特)
    received_pairs = received_bits.reshape(-1, 2)
    
    # 动态规划：路径度量和路径跟踪
    # path_metric[state] = 到当前状态的最小累计度量
    # paths[t, state] = 第 t 时刻到达 state 时的前一状态
    path_metric = np.full(num_states, np.inf, dtype=float)
    path_metric[0] = 0  # 初始状态为 0
    paths = np.zeros((num_steps, num_states), dtype=int)
    
    # Viterbi 轨迹
    for t, received_pair in enumerate(received_pairs):
        new_metric = np.full(num_states, np.inf, dtype=float)
        new_paths = np.zeros(num_states, dtype=int)
        
        # 对每个当前可达状态
        for curr_state in range(num_states):
            if path_metric[curr_state] == np.inf:
                continue
                
            # 对每个可能的输入（0 或 1）
            for input_bit in [0, 1]:
                # 计算下一个状态和输出
                # state = (s1, s0), 新下一个状态 = (input_bit, s1)
                next_state = (input_bit << 1) | ((curr_state >> 1) & 1)
                
                # 根据 g1, g2 计算输出
                s1 = (curr_state >> 1) & 1
                s0 = curr_state & 1
                out1 = input_bit ^ s1 ^ s0  # g1 = 111
                out2 = input_bit ^ s0       # g2 = 101
                
                # 计算汉明距离
                hamming_dist = int((out1 != received_pair[0]) + (out2 != received_pair[1]))
                new_metric_value = path_metric[curr_state] + hamming_dist
                
                # 更新最小路径
                if new_metric_value < new_metric[next_state]:
                    new_metric[next_state] = new_metric_value
                    new_paths[next_state] = curr_state
        
        path_metric = new_metric
        paths[t] = new_paths
    
    # 回溯：从最优终状态出发
    final_state = np.argmin(path_metric)
    
    # 追踪轨迹
    decoded_bits = []
    state = final_state
    for t in range(num_steps - 1, -1, -1):
        prev_state = paths[t, state]
        # 从 prev_state 到 state 的输入比特是 input_bit
        # state = (input_bit << 1) | ((prev_state >> 1) & 1)
        input_bit = (state >> 1) & 1
        decoded_bits.insert(0, input_bit)
        state = prev_state
    
    # 删除最后 2 个尾比特
    return np.array(decoded_bits[:-2], dtype=int)


def run_coding_demo():
    """运行 Part 1 演示并生成 BER 曲线。"""
    print('=' * 60)
    print('Part 1：信道编码实验')
    print('=' * 60)

    error_probabilities = np.array([0.001, 0.003, 0.01, 0.03, 0.06, 0.1])
    uncoded_ber = []
    coded_ber = []

    try:
        bits = generate_bits(4000, seed=2026)
        bits = bits[: len(bits) // 4 * 4]
        encoded = hamming74_encode(bits)

        for index, probability in enumerate(error_probabilities):
            uncoded_rx = binary_symmetric_channel(bits, probability, seed=100 + index)
            encoded_rx = binary_symmetric_channel(encoded, probability, seed=200 + index)
            decoded = hamming74_decode(encoded_rx)
            uncoded_ber.append(calculate_ber(bits, uncoded_rx))
            coded_ber.append(calculate_ber(bits, decoded))

        plot_ber_curve(
            error_probabilities,
            {'未编码': uncoded_ber, 'Hamming(7,4)': coded_ber},
            'Hamming(7,4) 编码前后 BER 对比',
            'coding_ber_curve.png',
        )
        print('✅ 已生成 results/coding_ber_curve.png')
    except NotImplementedError as error:
        print(f'⏸️ 尚未完成核心函数：{error}')
    except Exception as error:
        print(f'❌ Part 1 运行失败：{error}')


if __name__ == '__main__':
    run_coding_demo()
