"""Tests for the deterministic financial layer.

This is the part of the pipeline that must not be wrong, so it is the part with tests.
"""

from decimal import Decimal

from src.schemas import LineItem
from src.tools import (
    check_line_item_arithmetic,
    check_line_items_sum_to_subtotal,
    check_total_reconciles,
    expected_total,
    parse_money,
    sum_line_items,
)


def test_parse_money_plain():
    assert parse_money("12.50") == Decimal("12.50")
    assert parse_money("$1,234.56") == Decimal("1234.56")
    assert parse_money("  99 ") == Decimal("99")


def test_parse_money_thousands_separator():
    # CORD is Indonesian: '12.000' is twelve thousand, not twelve.
    assert parse_money("12.000") == Decimal("12000")
    assert parse_money("1.234.567") == Decimal("1234567")
    # European style, comma as decimal point
    assert parse_money("1.234,56") == Decimal("1234.56")


def test_parse_money_negatives_and_junk():
    assert parse_money("(123.45)") == Decimal("-123.45")
    assert parse_money("-50.00") == Decimal("-50.00")
    assert parse_money("n/a") is None
    assert parse_money("") is None
    assert parse_money(None) is None


def test_parse_money_never_uses_float_arithmetic():
    # 0.1 + 0.2 != 0.3 in binary floating point. It must here.
    assert parse_money("0.1") + parse_money("0.2") == parse_money("0.3")


def test_sum_line_items_prefers_stated_total():
    items = [
        LineItem(name="coffee", quantity=Decimal("2"), unit_price=Decimal("3.50")),
        LineItem(name="cake", quantity=Decimal("1"), unit_price=Decimal("4.25")),
    ]
    assert sum_line_items(items) == Decimal("11.25")


def test_expected_total_applies_tax_service_and_discount():
    assert expected_total(Decimal("100"), Decimal("10"), Decimal("5"), Decimal("2")) == Decimal(
        "113.00"
    )
    assert expected_total(None) is None


def test_total_that_reconciles_passes():
    finding = check_total_reconciles(
        Decimal("100.00"), Decimal("10.00"), None, None, Decimal("110.00")
    )
    assert finding.passed


def test_total_that_does_not_reconcile_is_caught():
    # The case that matters: the receipt claims a total its own components do not support.
    finding = check_total_reconciles(
        Decimal("100.00"), Decimal("10.00"), None, None, Decimal("150.00")
    )
    assert not finding.passed
    assert finding.computed["expected_total"] == "110.00"
    assert finding.computed["stated_total"] == "150.00"


def test_line_items_not_summing_to_subtotal_is_caught():
    items = [LineItem(name="x", quantity=Decimal("1"), unit_price=Decimal("5.00"))]
    finding = check_line_items_sum_to_subtotal(items, Decimal("99.00"))
    assert not finding.passed


def test_line_item_arithmetic_mismatch_is_caught():
    items = [
        LineItem(
            name="bad row",
            quantity=Decimal("3"),
            unit_price=Decimal("2.00"),
            total_price=Decimal("7.00"),  # should be 6.00
        )
    ]
    finding = check_line_item_arithmetic(items)
    assert not finding.passed
    assert finding.computed["mismatch_count"] == "1"


def test_missing_data_abstains_rather_than_failing():
    # Absence of data must not be reported as a failed check.
    assert check_total_reconciles(None, None, None, None, None).passed
    assert check_line_items_sum_to_subtotal([], None).passed
