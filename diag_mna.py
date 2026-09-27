import sys; sys.path.insert(0,'.')
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
from overwatcher.models import OverwatcherRequest, ProposedAction, OperationType, ResourceType, TrustProfile
import base64, hashlib, hmac

FAKE_TRUST = TrustProfile(agent_id='a', trust_score=0.85, total_actions=500, total_violations=0, last_updated=datetime.now(timezone.utc))
mt = MagicMock(); mt.get_trust_profile.return_value=FAKE_TRUST; mt.reward.return_value=FAKE_TRUST; mt.penalize.return_value=FAKE_TRUST
ma = MagicMock(); ma.log.return_value='x'

def jwt(role):
    h = base64.urlsafe_b64encode(b'{"alg":"HS256","typ":"JWT"}').rstrip(b'=').decode()
    p = base64.urlsafe_b64encode(f'{{"sub":"dr","role":"{role}","exp":9999999999}}'.encode()).rstrip(b'=').decode()
    s = hmac.new(b'final_year_project_overwatcher', f'{h}.{p}'.encode(), hashlib.sha256).digest()
    return f'{h}.{p}.{base64.urlsafe_b64encode(s).rstrip(b"=").decode()}'

with patch('overwatcher.modules.trust_memory.memory.TrustMemory', return_value=mt), \
     patch('overwatcher.modules.audit_logger.logger.AuditLogger', return_value=ma):
    from overwatcher.pipeline import OverwatcherPipeline
    pl = OverwatcherPipeline()
pl._trust_memory=mt; pl._audit=ma

tests = [
    ('Get blood pressure systolic diastolic and pulse rate readings for patient John',
     OperationType.READ, ResourceType.LAB_REPORT,
     ['blood pressure', 'systolic reading', 'diastolic reading', 'pulse rate'], 'John', 'doctor'),
    ('List medication name dosage prescription date prescribing doctor and refill count for patient John',
     OperationType.READ, ResourceType.PRESCRIPTION,
     ['medication name', 'dosage', 'prescription date', 'prescribing doctor', 'refill count'], 'John', 'doctor'),
    ('Show appointment schedule with date time doctor name patient name and appointment type',
     OperationType.READ, ResourceType.APPOINTMENT,
     ['appointment date', 'appointment time', 'doctor name', 'patient name', 'appointment type'], None, 'doctor'),
    ('Retrieve CBC blood count WBC RBC hemoglobin and platelet results for patient Mary',
     OperationType.READ, ResourceType.LAB_REPORT,
     ['CBC result', 'white blood cell count', 'red blood cell count', 'hemoglobin level', 'platelet count'], 'Mary', 'doctor'),
]

for raw, op, res, fields, pid, role in tests:
    r = pl.run(OverwatcherRequest(
        session_id='diag-x',
        raw_request=raw,
        proposed_action=ProposedAction(
            agent_id='a', operation=op, resource_type=res,
            patient_id=pid, recipients=[], data_fields=fields,
            raw_tool_call={}),
        auth_token=jwt(role), device_id='HOSP-DEV-001'))
    print(f"Request : {raw[:60]}")
    print(f"Decision: {r.decision.value}  Rule: {r.policy_result.rule_fired}")
    print(f"Intent  : op={r.parsed_intent.operation.value}  res={r.parsed_intent.resource_type.value}  conf={r.parsed_intent.confidence}")
    print(f"Consist : ok={r.consistency_result.is_consistent}  sim={r.consistency_result.similarity_score}")
    for v in r.consistency_result.violations:
        print(f"  VIOLATION: {v}")
    print()
