def tokenizer(s: str) -> list[str]:
    state = 'NORM'
    tokens = []
    token = ''
    for c in s:
        if state == 'NORM' and c == '\\':
            state = 'ESCAPE'
        elif state == 'ESCAPE':
            tokens.append(c)
            state = 'NORM'
        elif state == 'SPECIAL':
            if c == '>':
                tokens.append(token + c)
                state = 'NORM'
            else:
                token += c
        elif c == '<':
            state = 'SPECIAL'
            token = c
        else:
            tokens.append(c)
    return tokens


class ST:
    PAD = '<PAD>'
    GO = '<GO>'
    EOS = '<STOP>'
    THINK = '<THINK>'


class Label:
    PAD = 0
    Q = 1
    T = 2
    A = 3
