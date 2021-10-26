
import numpy as np


pad = '<pad>'
eos = '<eos>'
mask = [f'm_{num}' for num in range(1)]

special_tokens = [pad, eos]

time_signature_token = ['4/4', '3/4', '2/4', '6/8']

program_num = [f'i_{num}' for num in range(128)]

tempo_token = [f't_{i}' for i in range(7)]

track_num = [f'track_{num}' for num in range(3)]

structure_token = ['bar'] + track_num

song_token = time_signature_token + tempo_token + program_num


step_token = [f'e_{num}' for num in range(16)]

duration_single = [f'n_{num}' for num in range(1,33)]

pitch_tokens = [f'p_{num}' for num in range(21, 109)]


all_key_names = ['C major', 'G major', 'D major', 'A major',
                 'E major', 'B major', 'F major', 'B- major',
                 'E- major', 'A- major', 'D- major', 'G- major',
                 'A minor', 'E minor', 'B minor', 'F# minor',
                 'C# minor', 'G# minor', 'D minor', 'G minor',
                 'C minor', 'F minor', 'B- minor', 'E- minor',
                 ]

all_major_names = np.array(['C major', 'D- major', 'D major', 'E- major',
                   'E major', 'F major', 'G- major', 'G major',
                   'A- major', 'A major', 'B- major', 'B major'])

all_minor_names = np.array(['A minor', 'B- minor', 'B minor', 'C minor',
                   'C# minor', 'D minor', 'E- minor', 'E minor',
                   'F minor', 'F# minor', 'G minor', 'G# minor'])


key_token = [f'k_{num}' for num in range(len(all_key_names))]
key_to_token = {name: f'k_{i}' for i, name in enumerate(all_key_names)}
token_to_key = {v: k for k, v in key_to_token.items()}


track_note_density_token = [f'd_{num}' for num in range(10)]
track_occupation_rate_token = [f'o_{num}' for num in range(10)]
track_polyphony_rate_token = [f'y_{num}' for num in range(10)]
track_pitch_register_token = [f'r_{num}' for num in range(8)]


tensile_strain_token = [f's_{num}' for num in range(12)]
diameter_token = [f'a_{num}' for num in range(12)]
tension_token = [f'l_{num}' for num in range(12)]


control_bins = np.arange(0, 1, 0.1)
tensile_bins = np.arange(0, 2.1, 0.2).tolist() + [4]
diameter_bins = np.arange(0, 4.1, 0.4).tolist() + [5]

tempo_bins = np.array([0] + list(range(60, 190, 30)) + [200])
tension_bin = np.arange(0,6.5,0.5)
tension_bin[-1] = 6.5

class WordVocab(object):
    def __init__(self, mode,control_list):
        super(WordVocab, self).__init__()
        #control list is []


        duration_only_token = duration_single
        duration_tokens = step_token + duration_only_token


        note_tokens = pitch_tokens + duration_tokens

        basic_tokens = special_tokens + mask + structure_token + \
                     song_token + note_tokens

        all_tokens = basic_tokens + \
                     track_note_density_token + \
                     track_polyphony_rate_token + \
                     track_occupation_rate_token + \
                     key_token + tensile_strain_token + \
                     diameter_token


        self.pad_index = 0
        self.eos_index = 1
        self.char_lst = all_tokens
        self.basic_tokens = basic_tokens

        self._char2idx = {
            '<pad>': self.pad_index,
            '<eos>': self.eos_index,
        }

        for char in self.char_lst:
            if char not in self._char2idx:
                self._char2idx[char] = len(self._char2idx)
        self._idx2char = dict((idx, char) for char, idx in self._char2idx.items())
        print(f'vocab size: {self.vocab_size}')

        self.token_class_ranges = {}
        self.name_to_tokens = {}
        self.structure_indices = [self._char2idx[name] for name in structure_token]
        self.pitch_indices = [self._char2idx[name] for name in pitch_tokens]
        self.mask_indices = [self._char2idx[name] for name in mask]
        self.duration_indices = [self._char2idx[name] for name in duration_tokens]
        self.duration_only_indices = [self._char2idx[name] for name in duration_only_token]
        self.program_indices = [self._char2idx[name] for name in program_num]
        self.tempo_indices = [self._char2idx[name] for name in tempo_token]
        self.time_signature_indices = [self._char2idx[name] for name in time_signature_token]
        self.rest_indices = []
        self.control_indices = {}
        self.control_tokens = []





        self.step_indices = [self._char2idx[name] for name in step_token]

        for index in self.program_indices:
            self.token_class_ranges[index] = 'program'
            if 'program' in self.name_to_tokens:
                self.name_to_tokens['program'].append(self._idx2char[index])
            else:
                self.name_to_tokens['program'] = [self._idx2char[index]]

        for index in self.rest_indices:
            self.token_class_ranges[index] = 'rests'
            if 'rests' in self.name_to_tokens:
                self.name_to_tokens['rests'].append(self._idx2char[index])
            else:
                self.name_to_tokens['rests'] = [self._idx2char[index]]
        for index in self.tempo_indices:
            self.token_class_ranges[index] = 'tempo'
            if 'tempo' in self.name_to_tokens:
                self.name_to_tokens['tempo'].append(self._idx2char[index])
            else:
                self.name_to_tokens['tempo'] = [self._idx2char[index]]
        for index in self.time_signature_indices:
            self.token_class_ranges[index] = 'time_signature'
            if 'time_signature' in self.name_to_tokens:
                self.name_to_tokens['time_signature'].append(self._idx2char[index])
            else:
                self.name_to_tokens['time_signature'] = [self._idx2char[index]]

        for index in self.structure_indices:
            self.token_class_ranges[index] = 'structure'
            if 'structure' in self.name_to_tokens:
                self.name_to_tokens['structure'].append(self._idx2char[index])
            else:
                self.name_to_tokens['structure'] = [self._idx2char[index]]
        for index in self.pitch_indices:
            self.token_class_ranges[index] = 'pitch'
            if 'pitch' in self.name_to_tokens:
                self.name_to_tokens['pitch'].append(self._idx2char[index])
            else:
                self.name_to_tokens['pitch'] = [self._idx2char[index]]
        for index in self.duration_indices:
            self.token_class_ranges[index] = 'duration'
            if 'duration' in self.name_to_tokens:
                self.name_to_tokens['duration'].append(self._idx2char[index])
            else:
                self.name_to_tokens['duration'] = [self._idx2char[index]]

        self.token_class_ranges[self.eos_index] = 'eos'

        self.name_to_tokens['eos'] = self._idx2char[self.eos_index]


        if 'key' in control_list:
            self.key_indices = [self._char2idx[name] for name in key_token]
            self.control_indices['key'] = self.key_indices


            for index in self.key_indices:
                self.token_class_ranges[index] = 'key'
                if 'key' in self.name_to_tokens:
                    self.name_to_tokens['key'].append(self._idx2char[index])
                else:
                    self.name_to_tokens['key'] = [self._idx2char[index]]

            self.control_tokens.extend(self.name_to_tokens['key'])

        if 'density' in control_list:
            self.density_indices = [self._char2idx[name] for name in track_note_density_token]
            self.control_indices['density'] = self.density_indices
            for index in self.density_indices:
                self.token_class_ranges[index] = 'density'
                if 'density' in self.name_to_tokens:
                    self.name_to_tokens['density'].append(self._idx2char[index])
                else:
                    self.name_to_tokens['density'] = [self._idx2char[index]]
            self.control_tokens.extend(self.name_to_tokens['density'])

        if 'occupation' in control_list:
            self.occupation_indices = [self._char2idx[name] for name in track_occupation_rate_token]
            self.control_indices['occupation'] = self.occupation_indices

            for index in self.occupation_indices:
                self.token_class_ranges[index] = 'occupation'
                if 'occupation' in self.name_to_tokens:
                    self.name_to_tokens['occupation'].append(self._idx2char[index])
                else:
                    self.name_to_tokens['occupation'] = [self._idx2char[index]]

            self.control_tokens.extend(self.name_to_tokens['occupation'])

        if 'polyphony' in control_list:
            self.polyphony_indices = [self._char2idx[name] for name in track_polyphony_rate_token]
            self.control_indices['polyphony'] = self.polyphony_indices

            for index in self.polyphony_indices:
                self.token_class_ranges[index] = 'polyphony'
                if 'polyphony' in self.name_to_tokens:
                    self.name_to_tokens['polyphony'].append(self._idx2char[index])
                else:
                    self.name_to_tokens['polyphony'] = [self._idx2char[index]]

            self.control_tokens.extend(self.name_to_tokens['polyphony'])

        if 'tensile' in control_list:

            self.tensile_indices = [self._char2idx[name] for name in tensile_strain_token]
            self.control_indices['tensile'] = self.tensile_indices

            for index in self.tensile_indices:
                self.token_class_ranges[index] = 'tensile'
                if 'tensile' in self.name_to_tokens:
                    self.name_to_tokens['tensile'].append(self._idx2char[index])
                else:
                    self.name_to_tokens['tensile'] = [self._idx2char[index]]

            self.control_tokens.extend(self.name_to_tokens['tensile'])

        if 'diameter' in control_list:
            self.diameter_indices = [self._char2idx[name] for name in diameter_token]
            self.control_indices['diameter'] = self.diameter_indices

            for index in self.diameter_indices:
                self.token_class_ranges[index] = 'diameter'
                if 'diameter' in self.name_to_tokens:
                    self.name_to_tokens['diameter'].append(self._idx2char[index])
                else:
                    self.name_to_tokens['diameter'] = [self._idx2char[index]]
            self.control_tokens.extend(self.name_to_tokens['diameter'])

        self.class_names = set(self.token_class_ranges.values())

    def char2index(self, token):
        # print(token)
        if token not in self._char2idx:
            print('invalid')

        return self._char2idx.get(token)

    def index2char(self, idxs):

        return self._idx2char.get(idxs)

    def get_token_classes(self, idx):

        return self.token_class_ranges[idx]

    @property
    def vocab_size(self):
        return len(self._char2idx)

