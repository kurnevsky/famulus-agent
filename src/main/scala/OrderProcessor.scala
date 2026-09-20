package com.agent

import scala.util.{Either, Left, Right}

/** Represents a customer order. */
final case class Order(
    id: String,
    customerId: String,
    items: List[OrderItem],
    currency: String
) {
  def total: Int = items.map(i => i.unitPrice * i.quantity).sum // in cents
}

/** A single line item in an order. */
final case class OrderItem(
    sku: String,
    name: String,
    quantity: Int,
    unitPrice: Int // in cents
)

object OrderProcessor:

  /** Validation errors that can occur for an order. */
  sealed trait OrderError
  case object EmptyOrder extends OrderError
  case class NegativeQuantity(sku: String) extends OrderError
  case class InvalidCurrency(code: String) extends OrderError

  private val SupportedCurrencies: Set[String] =
    Set("USD", "EUR", "GBP", "CHF", "JPY")

  /** Validates an order, returning a list of all problems found. */
  def validate(order: Order): List[OrderError] =
    val errors = List.empty[OrderError]
    if order.items.isEmpty then errors :+ EmptyOrder
    else
      val badQty = order.items
        .filter(_.quantity <= 0)
        .map(i => NegativeQuantity(i.sku): OrderError)
      val badCur =
        if SupportedCurrencies.contains(order.currency)
        then Nil
        else List(InvalidCurrency(order.currency))
      errors ++ badQty ++ badCur

  /**
   * Processes an order: validates, then applies a discount.
   * Returns Either an error or the final total in cents.
   */
  def process(order: Order): Either[OrderError, Int] =
    val errors = validate(order)
    errors.headOption match
      case Some(err) => Left(err)
      case None      => Right(applyDiscount(order.total))

  /** Loyalty discount: 10% off orders above 100 000 cents (i.e., $1000). */
  def applyDiscount(total: Int): Int =
    if total > 100_000 then (total * 90) / 100
    else total

  def main(args: Array[String]): Unit =
    val sample = Order(
      id = "ord-001",
      customerId = "cust-42",
      items = List(
        OrderItem("SKU-1", "Mechanical Keyboard", 2, 149_99),
        OrderItem("SKU-2", "USB-C Cable", 5, 12_50)
      ),
      currency = "USD"
    )

    process(sample) match
      case Right(total) =>
        println(s"Order ${sample.id} approved. Total: ${total / 100}.${(total % 100) / 10} ${sample.currency}")
      case Left(err) =>
        println(s"Order ${sample.id} rejected: $err")

    val bad = sample.copy(items = Nil, currency = "XYZ")
    println(s"Bad order errors: ${validate(bad).mkString(", ")}")
