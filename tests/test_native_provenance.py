"""Native and Inspect readers must agree on explicit gameplay protocol evidence."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from delegated_influence import episodes, experiment, task


def sample(human_board=False):
    return task.make_episode_sample({'P1':'mockllm/a','P2':'mockllm/b'}, 1, 1, 41,
                                    'provenance', human_board=human_board)


@pytest.mark.parametrize('human_board,version', [(False,4), (True,5)])
def test_new_samples_stamp_the_gameplay_instrument(human_board, version):
    metadata=sample(human_board).metadata
    assert metadata['gameplay_protocol_instrument'] == {'status':'versioned','version':version}


@pytest.mark.parametrize('resolve_instruments', [False,True])
@pytest.mark.parametrize('version', [2,3,None])
def test_older_native_exports_recover_only_their_stored_version(tmp_path, resolve_instruments, version):
    metadata=deepcopy(sample().metadata)
    metadata.pop('gameplay_protocol_instrument', None)
    if version is None:
        metadata.pop('gameplay_protocol_version')
    else:
        metadata['gameplay_protocol_version']=version
    saved={'schema_version':'native-episode-v1','status':'complete','episode_id':'native-id',
           'sample_id':'provenance','metadata':metadata,
           'record':{**deepcopy(metadata),'events':[],'scores':{'P1':0,'P2':0}}}
    file=tmp_path/'one.episode.json'
    file.write_text(json.dumps(saved))
    row=episodes.load_episode_rows([file], resolve_instruments=resolve_instruments)[0]
    expected=experiment.infer_gameplay_protocol(SimpleNamespace(metadata=metadata,events=[]))
    assert row['metadata']['gameplay_protocol_instrument'] == expected
    assert row['record']['gameplay_protocol_instrument'] == expected
    comparable={**metadata,'gameplay_protocol_instrument':expected}
    assert experiment.legacy_cohort_identity(row['metadata']) == experiment.legacy_cohort_identity(comparable)
    assert json.loads(file.read_text()) == saved  # Recovery never rewrites historical evidence.
