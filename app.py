import logging
logging.basicConfig(level=logging.DEBUG)

from flask import Flask, render_template, request, jsonify, send_from_directory
from chessbot import Engine
import chess
import time
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable caching

engine = None
current_hash = None

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/new_game', methods=['POST'])
def new_game():
    global engine, current_hash
    data = request.get_json() or {}
    fen = data.get('fen')
    
    try:
        engine = Engine(fen)
        current_hash = engine.zobrist_hash()
        return jsonify({
            'success': True,
            'fen': engine.board.fen(),
            'turn': 'white' if engine.board.turn else 'black'
        })
    except Exception as e:
        print(f"Error in new_game: {e}")
        engine = Engine()
        current_hash = engine.zobrist_hash()
        return jsonify({
            'success': False,
            'error': str(e),
            'fen': engine.board.fen()
        })

@app.route('/make_move', methods=['POST'])
def make_move():
    global engine, current_hash
    data = request.get_json() or {}
    move_uci = data.get('move')
    
    if not move_uci:
        return jsonify({'success': False, 'error': 'No move provided'})
    
    if engine is None:
        engine = Engine()
        current_hash = engine.zobrist_hash()
    
    try:
        move = chess.Move.from_uci(move_uci)
        if move in engine.board.legal_moves:
            current_hash = engine.push(move, current_hash)
            return jsonify({
                'success': True, 
                'fen': engine.board.fen()
            })
        return jsonify({'success': False, 'error': 'Invalid move'})
    except Exception as e:
        print(f"Error in make_move: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/get_move', methods=['POST'])
def get_move():
    global engine, current_hash
    try:
        data = request.get_json() or {}
        depth = int(data.get('depth', 3))
        time_limit = float(data.get('time_limit', 2)) * 60  # Convert to seconds
        
        if engine is None:
            engine = Engine()
            current_hash = engine.zobrist_hash()
        
        # Check endgame condition - FIXED
        # The piece_map() method doesn't take arguments
        all_pieces = engine.board.piece_map()
        w_pieces = 0
        b_pieces = 0
        
        for square, piece in all_pieces.items():
            if piece.color == chess.WHITE:
                w_pieces += 1
            else:
                b_pieces += 1
                
        engine.endgame = 1 if min(w_pieces, b_pieces) <= 4 else 0
        
        start_time = time.time()
        result = engine.iterative_deepening(depth, current_hash, time_limit)
        
        if result:
            eval_score, best_move = result
            # Make the move on the engine
            current_hash = engine.push(best_move, current_hash)
            
            return jsonify({
                'success': True,
                'move': best_move.uci(),
                'evaluation': float(eval_score),
                'time': time.time() - start_time,
                'nodes': engine.nodes,
                'fen': engine.board.fen()
            })
        return jsonify({'success': False, 'error': 'No move found'})
    except Exception as e:
        print(f"Error in get_move: {e}")
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)