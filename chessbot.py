import chess
import random
import time
import tensorflow as tf
import numpy as np
import math
from tensorflow import keras
from tensorflow.keras.models import load_model
import onnx
import onnxruntime as ort
import numba as nb

class t_entry:
    def __init__(self, value, depth, best_move=None):
        self.value = value
        self.encoded_board = ''
        self.depth = depth
        self.best_move = best_move
        
model = ort.InferenceSession("NNeval.onnx", providers=["CPUExecutionProvider"])
        
def encode_board(board):
    matrix = np.zeros(64*13, dtype=np.float32)
    for i in range(64):
        piece = board.piece_at(i)
        if piece:
            matrix[13*i+piece.piece_type-1+(0 if piece.color ^ board.turn else 6)] = 1
        else:
            matrix[13*i+12] = 1
    return matrix

piece_value = {
    1: 100,
    2: 300,
    3: 300,
    4: 500,
    5: 900,
    6: 500,
}

class Engine:
    def __init__(self, FEN=None):
        global piece_value
        self.board = chess.Board(FEN) if FEN else chess.Board()
        self.z_table = tuple(random.getrandbits(64) for _ in range(781))
        self.t_table = {}
        self.r_table = {}
        self.nodes = 0
        self.killer_moves = [[] for _ in range(100)]
        self.endgame = 0

    def zobrist_hash(self):
        index = 0
        for square, piece in self.board.piece_map().items():
            index ^= self.z_table[(piece.piece_type-1)+(0 if piece.color else 6)+12*square]
        if self.board.ep_square:
            index ^= self.z_table[768+self.board.ep_square%8]
        if self.board.has_kingside_castling_rights(chess.WHITE):
            index ^= self.z_table[776]
        if self.board.has_queenside_castling_rights(chess.WHITE):
            index ^= self.z_table[777]
        if self.board.has_kingside_castling_rights(chess.BLACK):
            index ^= self.z_table[778]
        if self.board.has_queenside_castling_rights(chess.BLACK):
            index ^= self.z_table[779]
        if self.board.turn == chess.WHITE:
            index ^= self.z_table[780]
        return index

    def push(self, move, ind):
        temp = str(move)
        loc1 = chess.parse_square(temp[0:2])
        loc2 = chess.parse_square(temp[2:4])
        p1 = self.board.piece_at(loc1)
        ind ^= self.z_table[(p1.piece_type-1)+(0 if p1.color else 6)+12*loc1]
        if self.board.is_capture(move):
            if self.board.is_en_passant(move):
                if self.board.turn == chess.WHITE:
                    ind ^= self.z_table[6+12*(loc2-8)]
                else:
                    ind ^= self.z_table[12*(loc2+8)]
            else:
                p2 = self.board.piece_at(loc2)
                ind ^= self.z_table[(p2.piece_type-1)+(0 if p2.color else 6)+12*loc2]
        if move.promotion:
            ind ^= self.z_table[(move.promotion-1)+(0 if p1.color else 6)+12*loc2]
        else:
            ind ^= self.z_table[(p1.piece_type-1)+(0 if p1.color else 6)+12*loc2]
        if self.board.ep_square:
            ind ^= self.z_table[768+self.board.ep_square%8]
        if self.board.has_kingside_castling_rights(chess.WHITE):
            ind ^= self.z_table[776]
        if self.board.has_queenside_castling_rights(chess.WHITE):
            ind ^= self.z_table[777]
        if self.board.has_kingside_castling_rights(chess.BLACK):
            ind ^= self.z_table[778]
        if self.board.has_queenside_castling_rights(chess.BLACK):
            ind ^= self.z_table[779]
            
        self.board.push(move)
        
        if self.board.ep_square:
            ind ^= self.z_table[768+self.board.ep_square%8]
        if self.board.has_kingside_castling_rights(chess.WHITE):
            ind ^= self.z_table[776]
        if self.board.has_queenside_castling_rights(chess.WHITE):
            ind ^= self.z_table[777]
        if self.board.has_kingside_castling_rights(chess.BLACK):
            ind ^= self.z_table[778]
        if self.board.has_queenside_castling_rights(chess.BLACK):
            ind ^= self.z_table[779]
        ind ^= self.z_table[780]

        return ind

    def nn_eval(self):
        inp = encode_board(self.board)
        inp = np.expand_dims(inp, axis=0)  
        return model.run(None, {"input": inp})[0][0][0]*((1 if self.board.turn else -1))

    def move_score(self,move,depth,hash_move):
        if move == hash_move:
            return 1000000
        elif self.board.is_capture(move):
            if self.board.is_en_passant(move):
                vic = 100
                atk = 100
            else:
                vic = piece_value[self.board.piece_at(move.to_square).piece_type]
                atk = piece_value[self.board.piece_at(move.from_square).piece_type]
            return 100000+vic-atk
        elif move in self.killer_moves[depth]:
            return 90000 - self.killer_moves[depth].index(move)*1000
        else:
            return 0

    def quiescence(self, alpha, beta, depth, ind):
        entry = self.t_table.get(ind)
        if entry:
            value = entry.value
        else:
            value = self.nn_eval()
            self.t_table[ind] = t_entry(value, 0)
            
        if self.board.turn == chess.WHITE:
            if value>=beta:
                return value
            else:
                alpha = value
        else:
            if value<=alpha:
                return value
            else:
                beta = value

        moves = list(self.board.legal_moves)
        captures = []
        for move in moves:
            if self.board.is_capture(move):
                if self.board.is_en_passant(move):
                    vic = 100
                    atk = 100
                else:
                    vic = piece_value[self.board.piece_at(move.to_square).piece_type]
                    atk = piece_value[self.board.piece_at(move.from_square).piece_type]
                captures.append((move,vic-atk))
        captures.sort(key=lambda x: x[1], reverse=True)

        if self.board.turn == chess.WHITE:
            max_eval = alpha
            for move in captures:
                new_ind = self.push(move[0], ind)
                value = self.quiescence(alpha, beta, depth+1, new_ind)
                self.board.pop()
                if value > max_eval:
                    max_eval = value
                    best_move = move
                alpha = max(alpha, value)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = beta
            for move in captures:
                new_ind = self.push(move[0], ind)
                value = self.quiescence(alpha, beta, depth + 1, new_ind)
                self.board.pop()
                if value < min_eval:
                    min_eval = value
                    best_move = move
                beta = min(beta, value)
                if beta <= alpha:
                    break
            return min_eval

    def minimax(self, alpha, beta, depth, ind, return_move,time_limit,start_time,max_depth):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                return None
        entry = self.t_table.get(ind)
        hash_move = None
        if depth!=max_depth and self.r_table.get(ind)==1:
            value = 0
            self.t_table[ind] = t_entry(value, float('inf'))
            return value
        if entry:
            if entry.depth >= depth:
                if return_move:
                    return [entry.value,entry.best_move]
                return entry.value
            if entry.best_move:
                hash_move = entry.best_move

        if self.board.is_checkmate():
            value = -9999999-depth if self.board.turn else 9999999+depth
            self.t_table[ind] = t_entry(value, depth)
            return value

        if self.board.is_stalemate() or self.board.is_insufficient_material() or self.board.is_fifty_moves():
            value = 0
            self.t_table[ind] = t_entry(value, float('inf'))
            return value
            
        if depth == 0:
            self.nodes += 1
            if self.board.is_check() or any(self.board.is_capture(move) for move in self.board.legal_moves):
                value = self.quiescence(alpha, beta, 0, ind)
                self.t_table[ind] = t_entry(value, depth)
            else:
                value = self.nn_eval()
                self.t_table[ind] = t_entry(value, depth)
            return value

        best_move = None
        moves = list(self.board.legal_moves)
        moves = sorted(moves, key=lambda move: self.move_score(move, depth, hash_move), reverse=True)
        
        if self.board.turn == chess.WHITE:
            if depth >= 3 and not self.board.is_check() and self.endgame==0:
                self.board.push(chess.Move.null())
                new_ind = ind ^ self.z_table[780]
                score = self.minimax(beta - 1, beta, depth - 2, new_ind, False, time_limit, start_time,max_depth)
                self.board.pop()
                if score is None:
                    return None
                if score >= beta:
                    return beta
            max_eval = -float('inf')
            ct = 0
            for move in moves:
                nop = 1
                if nop == 1 and depth >= 3 and ct >= 3 and not self.board.is_capture(move) and move not in self.killer_moves[depth] and not self.board.gives_check(move) and not self.board.is_check() and not move.promotion and self.endgame == 0:
                    new_ind = self.push(move, ind)
                    #value = self.minimax(alpha, beta, depth - 2, new_ind, False, time_limit, start_time,max_depth)
                    value = self.minimax(alpha, beta, int(math.sqrt((len(moves) - ct) / len(moves)) * depth - 1), new_ind, False, time_limit, start_time,max_depth)
                    if value is None:
                        self.board.pop()
                        return None
                    if value > alpha:
                        value = self.minimax(alpha, beta, depth - 1, new_ind, False, time_limit, start_time,max_depth)
                else:
                    new_ind = self.push(move, ind)
                    value = self.minimax(alpha, beta, depth - 1, new_ind, False, time_limit, start_time,max_depth)
                self.board.pop()
                if value is None:
                    return None
                if value > max_eval:
                    max_eval = value
                    best_move = move
                alpha = max(alpha, value)
                if beta <= alpha:
                    if not self.board.is_capture(move):
                        if move not in self.killer_moves[depth]:
                            self.killer_moves[depth].insert(0, move)
                        if len(self.killer_moves[depth]) > 2:
                            self.killer_moves[depth] = self.killer_moves[depth][:2]
                    break
                ct += 1
            self.t_table[ind] = t_entry(max_eval, depth, best_move)
            if return_move:
                return [max_eval, best_move]
            return max_eval
        else:
            if depth >= 3 and not self.board.is_check() and self.endgame==0:
                self.board.push(chess.Move.null())
                new_ind = ind ^ self.z_table[780]
                score = self.minimax(alpha, alpha + 1, depth - 2, new_ind, False, time_limit, start_time,max_depth)
                self.board.pop()
                if score is None:
                    return None
                if score <= alpha:
                    return alpha
            min_eval = float('inf')
            ct = 0
            for move in moves:
                if depth >= 3 and ct >= 3 and not self.board.is_capture(move) and move not in self.killer_moves[depth] and not self.board.gives_check(move) and not self.board.is_check() and not move.promotion and self.endgame==0:
                    new_ind = self.push(move, ind)
                    #value = self.minimax(alpha, beta, depth - 2, new_ind, False, time_limit, start_time,max_depth)
                    value = self.minimax(alpha, beta, int(math.sqrt((len(moves) - ct) / len(moves)) * depth), new_ind, False, time_limit, start_time,max_depth)
                    if value is None:
                        self.board.pop()
                        return None
                    if value < beta:
                        value = self.minimax(alpha, beta, depth - 1, new_ind, False, time_limit, start_time,max_depth)
                else:
                    new_ind = self.push(move, ind)
                    value = self.minimax(alpha, beta, depth - 1, new_ind, False, time_limit, start_time,max_depth)
                self.board.pop()
                if value is None:
                    return None
                if value < min_eval:
                    min_eval = value
                    best_move = move
                beta = min(beta, value)
                if beta <= alpha:
                    if not self.board.is_capture(move):
                        if move not in self.killer_moves[depth]:
                            self.killer_moves[depth].insert(0, move)
                        if len(self.killer_moves[depth]) > 2:
                            self.killer_moves[depth] = self.killer_moves[depth][:2]
                    break
                ct += 1
            #if min_eval > 9999999:
                #print(min_eval)
            self.t_table[ind] = t_entry(min_eval, depth, best_move)
            if return_move:
                return [min_eval, best_move]
            return min_eval

    def iterative_deepening(self, max_depth, ind, time_limit):
        start_time = time.time()
        self.killer_moves = [[] for _ in range(100)]
        best_result = None
        for depth in range(1, max_depth + 1):
            self.nodes = 0
            output = self.minimax(-999999999, 999999999, depth, ind, True, time_limit, start_time, depth)
            if output is None:
                print("depth reached:", depth-1)
                break
            best_result = output
        return best_result
        
def play(time_limit,FEN=None):
    engine = Engine(FEN)
    ind = engine.zobrist_hash()
    Given_depth = int(input("search depth "))
    while True:
        print(engine.board)
        if engine.board.is_checkmate():
            print("black win") if engine.board.turn else print("white win")
            return
        if engine.board.is_stalemate() or engine.board.is_insufficient_material() or engine.board.can_claim_draw():
            print("draw")
            return
        depth = Given_depth
        start = time.time()
        b_pieces = 0
        w_pieces = 0
        if engine.endgame == 0:
            for piece in engine.board.piece_map().values():
                if piece.color == chess.WHITE:
                    w_pieces+=1
                else:
                    b_pieces+=1
            if min(w_pieces,b_pieces)<=4:
                engine.endgame = 1
        #print(engine.minimax(-999999999, 999999999, depth, ind, True, time_limit, time.time()))
        output = engine.iterative_deepening(depth,ind,time_limit)
        end = time.time()
        val = output[0]
        move = output[1]
        print("evaluation:", val)
        print("best move:", move)
        print("nodes searched:", engine.nodes)
        print("time to search:", end-start)
        makemove = chess.Move.from_uci(input("make move"))
        if makemove in engine.board.legal_moves:
            ind = engine.push(makemove,ind)
        else:
            print("move invalid")


play(5000)