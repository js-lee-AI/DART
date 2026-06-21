from dart import SCRouter, SCRouteConfig, SCBudgetRouter, SCBudgetConfig, MockModelClient
EASY = 'What is 2 + 2?'
HARD = 'Compute the 7th term of a tricky recurrence.'
answers = {EASY: ['4', '4'], HARD: ['11', '13']}
print('== Stage 1: SC-Route ==')
router = SCRouter(MockModelClient(answers, think_answer='12'), SCRouteConfig(task_type='math'))
for q in (EASY, HARD):
    r = router.route(q)
    print(f'  {q!r:50s} -> {r.action:22s} answer={r.answer} think_tokens={r.thinking_tokens}')
print('\n== Stage 1+2: SC-Budget (budget probing) ==')
brouter = SCBudgetRouter(MockModelClient(answers, think_answer='12'), SCBudgetConfig(task_type='math', budget_stages=[1024, 2048]))
for q in (EASY, HARD):
    r = brouter.route(q)
    print(f'  {q!r:50s} -> {r.action:18s} answer={r.answer} budget={r.budget_stage_used} think_tokens={r.thinking_tokens}')
