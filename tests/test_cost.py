from jev_rag_bench.cost import ModelPrice, gemini_cost_usd, jev_cost_usd


def test_jev_cost_output_free():
    price = ModelPrice(input_usd_per_1m_tokens=0.042, output_usd_per_1m_tokens=0.0)
    cost = jev_cost_usd(price, input_tokens=1_000_000, output_tokens=1_000_000)
    assert cost == 0.042


def test_gemini_cost_includes_thinking_in_output():
    price = ModelPrice(input_usd_per_1m_tokens=1.25, output_usd_per_1m_tokens=10.0)
    cost = gemini_cost_usd(price, input_tokens=1_000_000, output_tokens=500_000, thinking_tokens=500_000)
    # output+thinking 합쳐서 1M 토큰 * $10/1M = $10, input 1M * $1.25/1M = $1.25
    assert cost == 1.25 + 10.0


def test_gemini_cost_zero_thinking():
    price = ModelPrice(input_usd_per_1m_tokens=0.30, output_usd_per_1m_tokens=2.50)
    cost = gemini_cost_usd(price, input_tokens=100, output_tokens=50, thinking_tokens=0)
    expected = (100 * 0.30 + 50 * 2.50) / 1_000_000
    assert abs(cost - expected) < 1e-12
