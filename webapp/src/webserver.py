#!/usr/bin/python
import random

import numpy
from bottle import route, run, abort, request, response, error, static_file
from json import dumps
import re

from webapp.src.evaluate import Evaluator

ROW_COUNT = 6
COLUMN_COUNT = 7
MAX_ITERS = 10000

ROW_PATTERN = re.compile(r"^[\.01]{" + str(COLUMN_COUNT) + "}")
evaluator = Evaluator()

@error(400)
@error(500)
def json_error(error):
    print(error)
    error_data = {
        'error': error.body
    }
    response.content_type = 'application/json'
    return dumps(error_data)


@route('/<filename:re:.*>')
def server_static(filename):
    return static_file(filename, root='../web')


@route('/connect4', method='POST')
def move():
    requestJson = request.json
    rows = requestJson['board']
    iter = requestJson['iters']

    state = numpy.array(convert_to_array(rows))
    result = evaluator.compute_action(state, -1)

    winner = result[0]
    action = result[1]

    response.content_type = 'application/json'

    return dumps({
        "move": action,
        "winner": winner
    })


def convert_to_array(rows):
    board = []
    for row in rows:
        int_row = []
        for index in range(len(row)):
            if row[index] == '*':
                int_val = -1
            else:
                int_val = int(row[index])
            int_row.append(int_val)
        board.append(int_row)
    return board


def populate_board(board, rows):
    for row in reversed(rows):
        for col, value in enumerate(row):
            if value != '.':
                lines = board.drop(col, value)
                if len(lines):
                    return value


run(host='localhost', port=8085, debug=True)
