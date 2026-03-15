ORGANS = [
    'Anus',
    'Bladder',
    'Levator ani muscle',
    'Pubis',
    'Rectum',
    'Urethra',
    'Uterus',
    'Vagina'
]

ORGAN_COL_NUM = 'organ_num'
ORGAN_COL = 'organ'

TARGET_PATH = './data/target_labels_clean.csv'
SEGMENTS_PATH = './data/segments'

PROLAPSE_TYPES = [
    'cystocele',
    'cystourethrocele',
    'uterine_prolapse',
    'cervical_elongation',
    'rectocele',
    'enterocele'
]

ANY_PROLAPSE = 'any_prolapse'
POSSIBLE_TARGETS = PROLAPSE_TYPES + [ANY_PROLAPSE]

