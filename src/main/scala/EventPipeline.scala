package com.example.pipeline

import scala.collection.mutable
import scala.concurrent.{ExecutionContext, Future}
import scala.concurrent.duration._
import scala.util.{Failure, Random, Success, Try}
import java.time.{Instant, LocalDate, ZoneId, ZonedDateTime}
import java.util.UUID

// ============================================================================
// Domain Model
// ============================================================================

/** Represents a unique identifier for any event in the pipeline. */
final case class EventId(value: String) {
  override def toString: String = s"event($value)"
}

object EventId {
  def random(): EventId = EventId(UUID.randomUUID().toString)

  private[domain] def parse(raw: String): Option[EventId] =
    if (raw.nonEmpty && raw.length <= 128) Some(EventId(raw)) else None
}

/** The kind of event flowing through the system. */
sealed trait EventType extends Product with Serializable

object EventType {
  case object PageView      extends EventType
  case object Click         extends EventType
  case object Purchase      extends EventType
  case object Refund        extends EventType
  case object Signup        extends EventType
  case object Logout        extends EventType
  case object Search        extends EventType
  case object Error         extends EventType
  case object HealthCheck   extends EventType

  val all: Seq[EventType] = Seq(
    PageView, Click, Purchase, Refund, Signup,
    Logout, Search, Error, HealthCheck
  )

  def fromString(raw: String): Option[EventType] = raw.toLowerCase match {
    case "pageview"    => Some(PageView)
    case "click"       => Some(Click)
    case "purchase"    => Some(Purchase)
    case "refund"      => Some(Refund)
    case "signup"      => Some(Signup)
    case "logout"      => Some(Logout)
    case "search"      => Some(Search)
    case "error"       => Some(Error)
    case "healthcheck" => Some(HealthCheck)
    case _             => None
  }
}

/** A user identifier wrapping an opaque string. */
final case class UserId(value: String) {
  override def toString: String = s"user($value)"
}

/** Monetary amount with a currency code. */
final case class Money(amount: BigDecimal, currency: String) {
  def +(other: Money): Money =
    if (currency == other.currency) Money(amount + other.amount, currency)
    else throw new IllegalArgumentException(
      s"Cannot add $currency and ${other.currency}")

  def negate: Money = Money(amount.negate, currency)
  def isZero: Boolean = amount.signum() == 0
}

object Money {
  val zero: Money = Money(BigDecimal(0), "USD")

  def of(amount: Double, currency: String = "USD"): Money =
    Money(BigDecimal(amount), currency)
}

/** Core event record flowing through the pipeline. */
final case class Event(
    id: EventId,
    kind: EventType,
    user: Option[UserId],
    timestamp: Instant,
    payload: EventPayload,
    metadata: Map[String, String],
    source: String
) {
  def withMetadata(key: String, value: String): Event =
    copy(metadata = metadata.updated(key, value))

  def age(at: Instant): FiniteDuration =
    (at.toEpochMilli - timestamp.toEpochMilli).millis

  def isUserEvent: Boolean = user.isDefined
}

object Event {
  def create(
      kind: EventType,
      user: Option[UserId],
      payload: EventPayload,
      metadata: Map[String, String],
      source: String,
      at: Instant = Instant.now()
  ): Event =
    Event(EventId.random(), kind, user, at, payload, metadata, source)
}

/** Sealed hierarchy of event payloads. */
sealed trait EventPayload extends Product with Serializable

final case class PageViewPayload(
    url: String,
    referrer: Option[String],
    durationMs: Long
) extends EventPayload

final case class ClickPayload(
    elementId: String,
    x: Int,
    y: Int,
    button: Int
) extends EventPayload

final case class PurchasePayload(
    orderId: String,
    items: Seq[OrderItem],
    total: Money,
    coupon: Option[String]
) extends EventPayload

final case class RefundPayload(
    orderId: String,
    amount: Money,
    reason: String
) extends EventPayload

final case class SignupPayload(
    channel: String,
    verified: Boolean
) extends EventPayload

final case class LogoutPayload(
    sessionLengthSec: Long
) extends EventPayload

final case class SearchPayload(
    query: String,
    resultCount: Int
) extends EventPayload

final case class ErrorPayload(
    code: String,
    message: String,
    stackTrace: Option[String]
) extends EventPayload

final case object HealthCheckPayload extends EventPayload

final case class OrderItem(
    sku: String,
    quantity: Int,
    unitPrice: Money
) {
  def lineTotal: Money =
    unitPrice.copy(amount = unitPrice.amount * quantity)
}

/** Result of running a validation rule against an event. */
final case class ValidationIssue(
    rule: String,
    severity: Severity,
    message: String
)

sealed trait Severity extends Product with Serializable
object Severity {
  case object Error   extends Severity
  case object Warning extends Severity
  case object Info    extends Severity

  def rank(s: Severity): Int = s match {
    case Error   => 0
    case Warning => 1
    case Info    => 2
  }
}

/** A processed event carrying its validation result. */
final case class ProcessedEvent(
    event: Event,
    issues: Seq[ValidationIssue],
    accepted: Boolean
) {
  def hasErrors: Boolean = issues.exists(_.severity == Severity.Error)
  def worstIssue: Option[ValidationIssue] =
    issues.sortBy(i => -Severity.rank(i.severity)).headOption
}

// ============================================================================
// Configuration
// ============================================================================

/** Immutable configuration for the whole pipeline. */
final case class PipelineConfig(
    maxBatchSize: Int,
    flushIntervalMs: Long,
    maxRetries: Int,
    backoffBaseMs: Long,
    backoffMaxMs: Long,
    windowSizeSec: Int,
    minSupport: Int,
    rateLimitPerMinute: Int,
    slowEventThresholdMs: Long,
    deduplicateWindowMs: Long,
    maxMetadataKeys: Int,
    allowedSources: Set[String],
    enableScoring: Boolean,
    scoreDecayHalfLifeSec: Double
)

object PipelineConfig {
  val default: PipelineConfig = PipelineConfig(
    maxBatchSize = 512,
    flushIntervalMs = 5000L,
    maxRetries = 5,
    backoffBaseMs = 100L,
    backoffMaxMs = 30000L,
    windowSizeSec = 30,
    minSupport = 3,
    rateLimitPerMinute = 100,
    slowEventThresholdMs = 2000L,
    deduplicateWindowMs = 60000L,
    maxMetadataKeys = 20,
    allowedSources = Set("web", "mobile", "partner", "internal"),
    enableScoring = true,
    scoreDecayHalfLifeSec = 3600.0
  )

  def parse(raw: Map[String, String]): Try[PipelineConfig] =
    Try {
      val cfg = default
      val withSize = raw.get("maxBatchSize").map(v => cfg.copy(maxBatchSize = v.toInt))
        .getOrElse(cfg)
      val withFlush = raw.get("flushIntervalMs").map(v => withSize.copy(flushIntervalMs = v.toLong))
        .getOrElse(withSize)
      val withRetries = raw.get("maxRetries").map(v => withFlush.copy(maxRetries = v.toInt))
        .getOrElse(withFlush)
      val withWindow = raw.get("windowSizeSec").map(v => withRetries.copy(windowSizeSec = v.toInt))
        .getOrElse(withRetries)
      val withLimit = raw.get("rateLimitPerMinute").map(v => withWindow.copy(rateLimitPerMinute = v.toInt))
        .getOrElse(withWindow)
      withLimit
    }
}

// ============================================================================
// Validation
// ============================================================================

/** A single validation rule. */
trait ValidationRule {
  def name: String
  def check(event: Event): Option[ValidationIssue]
}

final case class SourceRule(allowedSources: Set[String]) extends ValidationRule {
  val name = "source-allowed"

  def check(event: Event): Option[ValidationIssue] =
    if (allowedSources.contains(event.source)) None
    else Some(ValidationIssue(
      name,
      Severity.Error,
      s"Source '${event.source}' is not in the allowed set"
    ))
}

final case class TimestampRule(skewToleranceMs: Long) extends ValidationRule {
  val name = "timestamp-sane"

  def check(event: Event): Option[ValidationIssue] = {
    val now = Instant.now()
    val skew = Math.abs(now.toEpochMilli - event.timestamp.toEpochMilli)
    if (skew <= skewToleranceMs) None
    else Some(ValidationIssue(
      name,
      if (skew > 24 * 3600 * 1000L) Severity.Error else Severity.Warning,
      s"Timestamp skew of ${skew}ms exceeds tolerance"
    ))
  }
}

final case class MetadataSizeRule(maxKeys: Int) extends ValidationRule {
  val name = "metadata-size"

  def check(event: Event): Option[ValidationIssue] =
    if (event.metadata.size <= maxKeys) None
    else Some(ValidationIssue(
      name,
      Severity.Warning,
      s"Metadata has ${event.metadata.size} keys (max $maxKeys)"
    ))
}

final case class UserPresentRule extends ValidationRule {
  val name = "user-present"

  def check(event: Event): Option[ValidationIssue] =
    if (event.user.isDefined) None
    else if (event.kind == HealthCheck) None
    else Some(ValidationIssue(
      name,
      Severity.Warning,
      "User event without a user id"
    ))
}

final case class PurchaseTotalRule extends ValidationRule {
  val name = "purchase-total"

  def check(event: Event): Option[ValidationIssue] = event.payload match {
    case p: PurchasePayload =>
      val computed = p.items.foldLeft(Money.zero)(_ + _.lineTotal)
      if (computed.amount == p.total.amount) None
      else Some(ValidationIssue(
        name,
        Severity.Error,
        s"Purchase total ${p.total.amount} does not match computed ${computed.amount}"
      ))
    case _ => None
  }
}

final case class UrlRule extends ValidationRule {
  val name = "url-format"

  private val scheme = "^(https?|ftp)://[^\\s/$.?#].[^\\s]*$".r

  def check(event: Event): Option[ValidationIssue] = event.payload match {
    case p: PageViewPayload if !scheme.findFirstIn(p.url).isDefined =>
      Some(ValidationIssue(
        name,
        Severity.Error,
        s"Malformed URL '${p.url}'"
      ))
    case _ => None
  }
}

/** Runs a sequence of rules against an event and builds a ProcessedEvent. */
final class EventValidator(rules: Seq[ValidationRule]) {

  def validate(event: Event): ProcessedEvent = {
    val issues = rules.flatMap(_.check(event))
    val accepted = !issues.exists(_.severity == Severity.Error)
    ProcessedEvent(event, issues, accepted)
  }

  def validateAll(events: Seq[Event]): Seq[ProcessedEvent] =
    events.map(validate)
}

object EventValidator {
  def standard(config: PipelineConfig): EventValidator =
    new EventValidator(Seq(
      SourceRule(config.allowedSources),
      TimestampRule(skewToleranceMs = 10 * 60 * 1000L),
      MetadataSizeRule(config.maxMetadataKeys),
      UserPresentRule,
      PurchaseTotalRule,
      UrlRule
    ))
}

// ============================================================================
// Transformation
// ============================================================================

/** Normalizes event payloads and metadata for downstream consumers. */
object EventTransformer {

  def normalizeMetadata(meta: Map[String, String]): Map[String, String] =
    meta.map { case (k, v) =>
      val key = k.trim.toLowerCase.replace(' ', '_')
      val value = v.trim
      key -> value
    }.filter(_._2.nonEmpty)

  def redactUser(event: Event): Event =
    event.user match {
      case Some(uid) =>
        val hashed = java.security.MessageDigest.getInstance("SHA-256")
          .digest(uid.value.getBytes("UTF-8"))
          .map(b => f"$b%02x").mkString
        event.copy(user = Some(UserId(hashed.take16)))
      case None => event
    }

  def truncatePayload(event: Event): Event = event.payload match {
    case e: ErrorPayload =>
      event.copy(payload = e.copy(
        message = e.message.take(500),
        stackTrace = e.stackTrace.map(_.take(2048))
      ))
    case s: SearchPayload =>
      event.copy(payload = s.copy(query = s.query.take(200)))
    case other =>
      event.copy(payload = other)
  }

  def enrich(event: Event, at: Instant = Instant.now()): Event = {
    val enriched = event.metadata ++ Map(
      "processed_at" -> at.toString,
      "ingest_lag_ms" -> (at.toEpochMilli - event.timestamp.toEpochMilli).toString
    )
    event.copy(metadata = normalizeMetadata(enriched))
  }

  def transform(event: Event, redact: Boolean = true): Event = {
    var e = enrich(event)
    if (redact) e = redactUser(e)
    e = truncatePayload(e)
    e
  }
}

// ============================================================================
// Aggregation
// ============================================================================

/** Per-window aggregate counters for a user or source. */
final case class WindowCounters(
    windowStart: Instant,
    windowSize: FiniteDuration,
    counts: mutable.Map[String, Int] = mutable.Map.empty
) {
  def increment(key: String, by: Int = 1): Unit =
    counts(key) = counts.getOrElse(key, 0) + by

  def get(key: String): Int = counts.getOrElse(key, 0)

  def total: Int = counts.values.sum

  def snapshot: Map[String, Int] = counts.toMap

  def asMap: Map[String, Int] = counts.toMap
}

final class WindowCounter(windowSizeSec: Int) {
  private val size = windowSizeSec.seconds
  private val active = new WindowCounters(
    Instant.now().truncatedTo(java.time.temporal.ChronoUnit.SECONDS),
    size
  )
  private var current = active

  private def bucketFor(ts: Instant): Instant = {
    val ms = (ts.toEpochMilli / (windowSizeSec * 1000L)) * (windowSizeSec * 1000L)
    Instant.ofEpochMilli(ms)
  }

  def record(kind: EventType, at: Instant = Instant.now()): Unit = {
    val bucket = bucketFor(at)
    if (bucket != current.windowStart) {
      current = new WindowCounters(bucket, size)
    }
    current.increment(kind.toString.toLowerCase)
  }

  def currentCounters: WindowCounters = current

  def countFor(kind: EventType): Int = current.get(kind.toString.toLowerCase)
}

/** Computes rolling statistics over a bounded history. */
final class RollingStats(maxHistory: Int = 1024) {
  private val values = mutable.ArrayBuffer.empty[Double]
  private var sum: Double = 0.0
  private var sumSq: Double = 0.0
  private var min: Double = Double.MaxValue
  private var max: Double = Double.MinValue
  private var count: Long = 0

  def add(value: Double): Unit = {
    if (values.size >= maxHistory) {
      val old = values.removeAt(0)
      sum -= old
      sumSq -= old * old
      // Recompute min/max lazily; acceptable for approximate stats.
      min = values.min
      max = values.max
    }
    values += value
    sum += value
    sumSq += value * value
    if (value < min) min = value
    if (value > max) max = value
    count += 1
  }

  def mean: Double = if (count == 0) 0.0 else sum / count

  def variance: Double = {
    if (count < 2) 0.0
    else {
      val m = mean
      Math.max(0.0, (sumSq / count) - (m * m))
    }
  }

  def stddev: Double = math.sqrt(variance)

  def minV: Double = if (count == 0) 0.0 else min
  def maxV: Double = if (count == 0) 0.0 else max
  def total: Long = count

  def percentile(p: Double): Double = {
    if (count == 0) return 0.0
    val sorted = values.toSeq.sorted
    val idx = math.min(sorted.length - 1,
      math.max(0, (p * (sorted.length - 1)).toInt))
    sorted(idx)
  }
}

/** Aggregates purchases into revenue reports. */
object RevenueAggregator {

  final case class RevenueBucket(
      day: LocalDate,
      gross: Money,
      refunded: Money,
      orders: Int
  ) {
    def net: Money = gross.copy(amount = gross.amount - refunded.amount)
  }

  def aggregate(events: Seq[Event], currency: String = "USD"): Seq[RevenueBucket] = {
    val byDay = mutable.Map.empty[LocalDate, RevenueBucket]

    def add(kind: String, amount: Money, orderId: Option[String]): Unit => Unit = { _ =>
      ()
    }

    events.foreach { ev =>
      val day = ev.timestamp.atZone(ZoneId.of("UTC")).toLocalDate
      ev.payload match {
        case p: PurchasePayload =>
          val existing = byDay.getOrElse(day, RevenueBucket(day, Money.zero, Money.zero, 0))
          byDay(day) = existing.copy(
            gross = existing.gross + p.total,
            orders = existing.orders + 1
          )
        case r: RefundPayload =>
          val existing = byDay.getOrElse(day, RevenueBucket(day, Money.zero, Money.zero, 0))
          byDay(day) = existing.copy(
            refunded = existing.refunded + r.amount
          )
        case _ => ()
      }
    }
    byDay.values.toSeq.sortBy(_.day)
  }
}

// ============================================================================
// Windowing
// ============================================================================

/** A tumbling time window over event timestamps. */
final case class TimeWindow(
    start: Instant,
    end: Instant,
    events: Seq[Event] = Vector.empty
) {
  def contains(ts: Instant): Boolean =
    !ts.isBefore(start) && ts.isBefore(end)

  def size: Int = events.size

  def duration: FiniteDuration = (end.toEpochMilli - start.toEpochMilli).millis

  def merge(other: TimeWindow): TimeWindow =
    if (other.start == start) copy(events = events ++ other.events)
    else throw new IllegalArgumentException("Cannot merge windows with different starts")
}

final class TumblingWindower(windowSizeSec: Int) {

  def windowFor(ts: Instant): TimeWindow = {
    val size = windowSizeSec * 1000L
    val startMs = (ts.toEpochMilli / size) * size
    val start = Instant.ofEpochMilli(startMs)
    val end = Instant.ofEpochMilli(startMs + size)
    TimeWindow(start, end)
  }

  def assign(events: Seq[Event], at: Instant = Instant.now()): Map[Instant, Seq[Event]] =
    events.groupBy(windowFor(_).start).map { case (s, evts) => s -> evts }
}

final class SlidingWindower(
    windowSizeSec: Int,
    stepSizeSec: Int
) {
  require(stepSizeSec > 0 && stepSizeSec <= windowSizeSec, "Invalid sliding window config")

  def windowsFor(ts: Instant): Seq[TimeWindow] = {
    val startMs = ts.toEpochMilli
    val endMs = startMs
    val windows = Vector.newBuilder[TimeWindow]
    var offset = 0L
    val stepMs = stepSizeSec * 1000L
    val winMs = windowSizeSec * 1000L
    while (offset < winMs) {
      val s = endMs - offset - winMs + offset
      val wStart = endMs - winMs
      val wEnd = endMs - offset
      if (wStart < s) windows += TimeWindow(Instant.ofEpochMilli(wStart), Instant.ofEpochMilli(wEnd))
      offset += stepMs
    }
    windows.result()
  }
}

// ============================================================================
// Rate Limiting
// ============================================================================

/** Simple token bucket rate limiter. */
final class TokenBucket(
    capacity: Int,
    refillPerSecond: Double
) {
  private val rng = new Random(42)
  private var tokens: Double = capacity.toDouble
  private var lastRefill: Long = System.nanoTime()

  def tryAcquire(n: Int = 1): Boolean = synchronized {
    refill()
    if (tokens >= n) {
      tokens -= n
      true
    } else false
  }

  def available: Int = synchronized {
    refill()
    tokens.toInt
  }

  private def refill(): Unit = {
    val now = System.nanoTime()
    val elapsed = (now - lastRefill) / 1_000_000_000.0
    tokens = Math.min(capacity.toDouble, tokens + elapsed * refillPerSecond)
    lastRefill = now
  }
}

/** Sliding log rate limiter keyed by string. */
final class SlidingWindowLimiter(windowMs: Long, limit: Int) {
  private val hits = mutable.Map.empty[String, mutable.ArrayBuffer[Long]]

  def isAllowed(key: String, at: Long = System.currentTimeMillis()): Boolean = synchronized {
    val buf = hits.getOrElseUpdate(key, mutable.ArrayBuffer.empty)
    val cutoff = at - windowMs
    while (buf.nonEmpty && buf.head < cutoff) buf.removeFast()
    if (buf.size < limit) {
      buf += at
      true
    } else false
  }

  def remaining(key: String, at: Long = System.currentTimeMillis()): Int = synchronized {
    val buf = hits.getOrElse(key, mutable.ArrayBuffer.empty)
    val cutoff = at - windowMs
    val active = buf.count(_ >= cutoff)
    math.max(0, limit - active)
  }

  def reset(key: String): Unit = synchronized {
    hits.remove(key)
  }
}

// ============================================================================
// Retry and Backoff
// ============================================================================

/** Exponential backoff schedule with jitter. */
final class BackoffPolicy(
    baseMs: Long,
    maxMs: Long,
    factor: Double = 1.5,
    jitterRatio: Double = 0.2
) {
  private val rng = new Random()

  def delay(attempt: Int): Long = {
    require(attempt >= 0, "attempt must be non-negative")
    val raw = baseMs * math.pow(factor, attempt)
    val capped = math.min(raw, maxMs.toDouble).toLong
    val jitter = capped * jitterRatio * rng.nextDouble()
    (capped + jitter.toLong).max(0L)
  }

  def schedule: Iterator[Long] = Iterator.from(0).map(delay)
}

sealed trait AttemptResult[+A]
final case class Succeeded[A](value: A, attempts: Int) extends AttemptResult[A]
final case class FailedAfter[A](attempts: Int, lastError: Throwable) extends AttemptResult[A]

object Retry {
  def withBackoff[A](
      maxRetries: Int,
      policy: BackoffPolicy,
      sleep: Long => Unit = (ms: Long) => Thread.sleep(ms)
  )(action: () => Try[A]): AttemptResult[A] = {
    var attempt = 0
    var last: Option[Throwable] = None
    while (attempt <= maxRetries) {
      val result = try action() catch { case t: Throwable => Failure(t) }
      result match {
        case Success(v) => return Succeeded(v, attempt + 1)
        case Failure(t) =>
          last = Some(t)
          if (attempt < maxRetries) {
            sleep(policy.delay(attempt))
          }
      }
      attempt += 1
    }
    FailedAfter(attempts = attempt, lastError = last.getOrElse(new RuntimeException("no error")))
  }
}

// ============================================================================
// Deduplication
// ============================================================================

/** Deduplicates events by id within a sliding time window. */
final class Deduplicator(windowMs: Long) {
  private val seen = mutable.Map.empty[String, Long]

  def isDuplicate(id: String, at: Long = System.currentTimeMillis()): Boolean = synchronized {
    prune(at)
    val dup = seen.contains(id)
    seen(id) = at
    dup
  }

  def track(events: Seq[Event], at: Long = System.currentTimeMillis()): Seq[Event] =
    events.filterNot(e => isDuplicate(e.id.value, at))

  private def prune(at: Long): Unit = {
    val cutoff = at - windowMs
    val expired = seen.filter(_._2 < cutoff).keys.toSeq
    expired.foreach(seen.remove)
  }

  def size: Int = synchronized(seen.size)

  def clear(): Unit = synchronized(seen.clear())
}

// ============================================================================
// Scoring
// ============================================================================

/** Assigns a novelty score to events based on frequency and recency. */
final class EventScorer(halfLifeSec: Double, enable: Boolean = true) {
  private val frequencies = mutable.Map.empty[String, Int]
  private val lastSeen = mutable.Map.empty[String, Long]

  def score(event: Event, at: Instant = Instant.now()): Double = {
    if (!enable) return 1.0
    val key = event.kind.toString
    val freq = frequencies.getOrElse(key, 0) + 1
    frequencies(key) = freq
    val prev = lastSeen.get(key)
    lastSeen(key) = at.toEpochMilli
    val recency = prev match {
      case Some(t) =>
        val ageSec = (at.toEpochMilli - t) / 1000.0
        math.exp(-0.6931471805599453 * ageSec / halfLifeSec)
      case None => 1.0
    }
    val frequencyPenalty = 1.0 / math.sqrt(freq.toDouble)
    recency * frequencyPenalty
  }
}

// ============================================================================
// Metrics
// ============================================================================

/** In-memory metrics registry. */
final class MetricsRegistry {
  private val counters = mutable.Map.empty[String, Long]
  private val gauges = mutable.Map.empty[String, () => Double]
  private val histograms = mutable.Map.empty[String, RollingStats]

  def counter(name: String): Long = synchronized(counters.getOrElse(name, 0L))

  def increment(name: String, by: Long = 1L): Unit = synchronized {
    counters(name) = counters.getOrElse(name, 0L) + by
  }

  def gauge(name: String, compute: () => Double): Unit = synchronized {
    gauges(name) = compute
  }

  def record(name: String, value: Double): Unit = synchronized {
    val stats = histograms.getOrElseUpdate(name, new RollingStats(4096))
    stats.add(value)
  }

  def stats(name: String): Option[RollingStats] = synchronized(histograms.get(name))

  def snapshot: Map[String, Long] = synchronized(counters.toMap)

  def reset(): Unit = synchronized {
    counters.clear()
    gauges.clear()
    histograms.clear()
  }
}

/** Tracks end-to-end pipeline latency. */
final class LatencyTracker(registry: MetricsRegistry) {
  def record(name: String, startedNanos: Long, finishedNanos: Long): Double = {
    val ms = (finishedNanos - startedNanos) / 1_000_000.0
    registry.record(name, ms)
    registry.increment(s"$name.count")
    ms
  }
}

// ============================================================================
// Storage Abstractions
// ============================================================================

/** Abstraction over an event sink. */
trait EventSink {
  def name: String
  def write(batch: Seq[Event]): Try[Int]
  def close(): Unit
}

/** In-memory sink useful for tests and dry runs; not thread-safe beyond synchronized ops. */
final class InMemorySink(override val name: String = "in-memory") extends EventSink {
  private val store = mutable.ListBuffer.empty[Event]

  def write(batch: Seq[Event]): Try[Int] = {
    store ++= batch
    Success(batch.size)
  }

  def all: Seq[Event] = store.toSeq

  def size: Int = store.size

  def close(): Unit = store.clear()
}

/** File-backed sink appending JSON lines. */
final class FileSink(path: String) extends EventSink {
  val name = s"file($path)"
  @volatile private var writer: Option[java.io.BufferedWriter] = None

  def write(batch: Seq[Event]): Try[Int] = Try {
    val w = writer.getOrElse {
      val w = new java.io.BufferedWriter(new java.io.FileWriter(path, true))
      writer = Some(w)
      w
    }
    batch.foreach(e => w.write(EventJson.encode(e) + "\n"))
    w.flush()
    batch.size
  }

  def close(): Unit = writer.foreach(_.close()); writer = None
}

/** Abstraction over an event source. */
trait EventSource {
  def name: String
  def poll(batchSize: Int): Try[Seq[Event]]
  def close(): Unit
}

/** Source that replays a fixed list of events. */
final class ReplaySource(events: Seq[Event], override val name: String = "replay")
    extends EventSource {
  private var pos = 0

  def poll(batchSize: Int): Try[Seq[Event]] = Try {
    if (pos >= events.length) Seq.empty
    else {
      val end = math.min(events.length, pos + batchSize)
      val chunk = events.slice(pos, end)
      pos = end
      chunk
    }
  }

  def close(): Unit = ()
}

/** Source that generates synthetic events for load testing. */
final class SyntheticSource(seed: Long = 7L, ratePerSec: Double = 10.0)
    extends EventSource {
  val name = s"synthetic($ratePerSec/s)"
  private val rng = new Random(seed)

  private val kinds: Seq[EventType] = EventType.all.take(8)
  private val sources = Seq("web", "mobile", "partner")
  private val urls = Seq(
    "https://example.com/home",
    "https://example.com/products",
    "https://example.com/cart",
    "https://example.com/checkout"
  )

  private def randomPayload(kind: EventType): EventPayload = kind match {
    case PageView =>
      PageViewPayload(urls(rng.nextInt(urls.length)), None, rng.nextInt(30000).toLong)
    case Click =>
      ClickPayload(s"btn-${rng.nextInt(50)}", rng.nextInt(1920), rng.nextInt(1080), 0)
    case Purchase =>
      val items = (1 to rng.nextInt(3) + 1).map { _ =>
        OrderItem(s"sku-${rng.nextInt(1000)}", rng.nextInt(3) + 1,
          Money.of(rng.nextDouble() * 100.0 + 1.0))
      }
      PurchasePayload(
        s"order-${rng.nextInt(1000000)}",
        items,
        items.foldLeft(Money.zero)(_ + _.lineTotal),
        if (rng.nextDouble() < 0.2) Some(s"COUPON${rng.nextInt(100)}") else None
      )
    case Search =>
      SearchPayload(s"term ${rng.nextInt(500)}", rng.nextInt(20))
    case Error =>
      ErrorPayload(s"E${rng.nextInt(100)}", "synthetic error", None)
    case Signup =>
      SignupPayload(sources(rng.nextInt(3)), rng.nextDouble() < 0.8)
    case Logout =>
      LogoutPayload(rng.nextInt(7200).toLong)
    case Refund =>
      RefundPayload(s"order-${rng.nextInt(1000000)}", Money.of(rng.nextDouble() * 50.0), "damaged")
    case _ =>
      HealthCheckPayload
  }

  def poll(batchSize: Int): Try[Seq[Event]] = Try {
    val count = math.max(1, (ratePerSec / 10.0).toInt) * batchSize / 10
    (1 to count).map { _ =>
      val kind = kinds(rng.nextInt(kinds.length))
      val user = if (rng.nextDouble() < 0.85) Some(UserId(s"u${rng.nextInt(10000)}")) else None
      Event.create(
        kind,
        user,
        randomPayload(kind),
        Map("synthetic" -> "true"),
        sources(rng.nextInt(sources.length)),
        Instant.now()
      )
    }
  }

  def close(): Unit = ()
}

// ============================================================================
// Batch Queue
// ============================================================================

/** Collects events into batches bounded by size or time. */
final class BatchQueue(maxSize: Int, flushIntervalMs: Long) {
  private val buffer = mutable.Queue.empty[Event]
  private var lastFlush: Long = System.currentTimeMillis()

  def offer(event: Event): Unit = synchronized {
    buffer.enqueue(event)
  }

  def readyToFlush(at: Long = System.currentTimeMillis()): Boolean = synchronized {
    buffer.size >= maxSize || (at - lastFlush) >= flushIntervalMs
  }

  def drain(at: Long = System.currentTimeMillis()): Seq[Event] = synchronized {
    val out = if (readyToFlush(at)) buffer.dequeueAll().toSeq else Seq.empty
    if (out.nonEmpty) lastFlush = at
    out
  }

  def pending: Int = synchronized(buffer.size)

  def clear(): Unit = synchronized(buffer.clear())
}

// ============================================================================
// JSON Encoding
// ============================================================================

/** Minimal JSON encoder for events (no external dependencies). */
object EventJson {

  private def escape(s: String): String =
    s.flatMap {
      case '"'  => "\\\""
      case '\\' => "\\\\"
      case '\n' => "\\n"
      case '\r' => "\\r"
      case '\t' => "\\t"
      case c    => c.toString
    }

  private def encPayload(p: EventPayload): String = p match {
    case PageViewPayload(url, ref, dur) =>
      s"""{"type":"pageview","url":"${escape(url)}","referrer":${ref.map(r => s""""${escape(r)}"""").getOrElse("null")},"duration_ms":$dur}"""
    case ClickPayload(id, x, y, btn) =>
      s"""{"type":"click","element":"${escape(id)}","x":$x,"y":$y,"button":$btn}"""
    case PurchasePayload(order, items, total, coupon) =>
      val itemsJson = items.map { i =>
        s"""{"sku":"${escape(i.sku)}","qty":${i.quantity},"price":${i.unitPrice.amount},"ccy":"${escape(i.unitPrice.currency)}"}"""
      }.mkString("[", ",", "]")
      s"""{"type":"purchase","order":"${escape(order)}","items":$itemsJson,"total":${total.amount},"ccy":"${escape(total.currency)}","coupon":${coupon.map(c => s""""${escape(c)}"""").getOrElse("null")}}"""
    case RefundPayload(order, amount, reason) =>
      s"""{"type":"refund","order":"${escape(order)}","amount":${amount.amount},"ccy":"${escape(amount.currency)}","reason":"${escape(reason)}"}"""
    case SignupPayload(channel, verified) =>
      s"""{"type":"signup","channel":"${escape(channel)}","verified":$verified}"""
    case LogoutPayload(sec) =>
      s"""{"type":"logout","session_sec":$sec}"""
    case SearchPayload(q, n) =>
      s"""{"type":"search","query":"${escape(q)}","results":$n}"""
    case ErrorPayload(code, msg, st) =>
      s"""{"type":"error","code":"${escape(code)}","message":"${escape(msg)}","stack":${st.map(s => s""""${escape(s)}"""").getOrElse("null")}}"""
    case HealthCheckPayload =>
      """{"type":"healthcheck"}"""
  }

  def encode(event: Event): String = {
    val meta = event.metadata.map { case (k, v) =>
      s""""${escape(k)}":"${escape(v)}""""
    }.mkString("{", ",", "}")
    val user = event.user.map(u => s""""${escape(u.value)}""").getOrElse("null")
    s"""{"id":"${escape(event.id.value)}","kind":"${event.kind.toString.toLowerCase}","user":$user,"ts":"${event.timestamp.toString}","source":"${escape(event.source)}","payload":${encPayload(event.payload)},"meta":$meta}"""
  }
}

// ============================================================================
// Pipeline Orchestrator
// ============================================================================

/** The main orchestrator wiring validation, transformation, and sinks. */
final class Pipeline(
    config: PipelineConfig,
    source: EventSource,
    sinks: Seq[EventSink],
    registry: MetricsRegistry = new MetricsRegistry
) {
  private val validator = EventValidator.standard(config)
  private val queue = new BatchQueue(config.maxBatchSize, config.flushIntervalMs)
  private val dedup = new Deduplicator(config.deduplicateWindowMs)
  private val limiter = new SlidingWindowLimiter(60000L, config.rateLimitPerMinute)
  private val latency = new LatencyTracker(registry)
  private var running: Boolean = false

  /** Runs one poll-dedup-validate-transform-write cycle. */
  def step(): Try[Int] = {
    val started = System.nanoTime()
    source.poll(config.maxBatchSize).flatMap { raw =>
      registry.increment("source.polls")
      registry.increment("source.events", raw.size.toLong)

      val deduped = dedup.track(raw)
      registry.increment("dedup.removed", (raw.size - deduped.size).toLong)

      val processed = validator.validateAll(deduped)
      registry.increment("validation.errors",
        processed.count(_.hasErrors).toLong)

      val accepted = processed.filter(_.accepted).map { pe =>
        registry.increment("validation.accepted")
        pe.event.withMetadata("pipeline", "ok")
      }
      val rateLimited = accepted.filter(limiter.isAllowed(_))
      registry.increment("ratelimit.dropped", (accepted.size - rateLimited.size).toLong)

      val transformed = rateLimited.map(e =>
        EventTransformer.transform(e, redact = true))

      val batch = queue.offerAll(transformed)
      val toWrite = queue.drain()

      if (toWrite.nonEmpty) {
        sinks.foreach { sink =>
          sink.write(toWrite) match {
            case Success(n) => registry.increment(s"sink.${sink.name}.written", n.toLong)
            case Failure(t) =>
              registry.increment(s"sink.${sink.name}.failed")
              registry.increment("sink.failures")
          }
        }
        registry.increment("pipeline.batches")
      }

      val elapsedMs = latency.record("pipeline.step", started, System.nanoTime())
      if (elapsedMs > config.slowEventThresholdMs) {
        registry.increment("pipeline.slow_steps")
      }
      Success(toWrite.size)
    }
  }

  /** Runs the pipeline for a fixed number of steps. */
  def runSteps(steps: Int): Unit = {
    running = true
    try {
      (1 to steps).foreach { _ =>
        if (running) step()
      }
    } finally {
      running = false
    }
  }

  def stop(): Unit = {
    running = false
    queue.clear()
  }

  def isRunning: Boolean = running
  def pendingEvents: Int = queue.pending
}

// Helper extension on BatchQueue used by Pipeline.offerAll.
object BatchQueue {
  implicit class QueueOps(private val q: BatchQueue) extends AnyVal {
    def offerAll(events: Seq[Event]): Seq[Event] = {
      events.foreach(q.offer)
      events
    }
  }
}

// ============================================================================
// Utilities
// ============================================================================

/** Small collection of misc helpers used throughout the pipeline. */
object Util {

  /** Chunks a sequence into lists of at most `size`. */
  def chunked[A](seq: Seq[A], size: Int): Seq[Seq[A]] = {
    require(size > 0, "size must be positive")
    seq.grouped(size).toSeq
  }

  /** Exponential moving average update. */
  def emaUpdate(prev: Double, next: Double, alpha: Double): Double =
    alpha * next + (1 - alpha) * prev

  /** Formats an instant as ISO-8601 UTC. */
  def isoUtc(instant: Instant): String =
    ZonedDateTime.ofInstant(instant, ZoneId.of("UTC")).toString

  /** Parses a duration string like "30s", "5m", "2h" into milliseconds. */
  def parseDurationMs(raw: String): Try[Long] = Try {
    val number = raw.init.toLong
    raw.last match {
      case 's' => number * 1000L
      case 'm' => number * 60000L
      case 'h' => number * 3600000L
      case 'd' => number * 86400000L
      case _   => number
    }
  }

  /** Simple string hash for consistent bucket assignment. */
  def bucket(key: String, buckets: Int): Int = {
    require(buckets > 0, "buckets must be positive")
    math.abs(key.hashCode % buckets)
  }

  /** Merges maps, preferring values from `right` on key conflicts. */
  def mergeMaps[A](left: Map[String, A], right: Map[String, A]): Map[String, A] =
    left ++ right

  /** Clamps a value to the inclusive range [lo, hi]. */
  def clamp(v: Int, lo: Int, hi: Int): Int =
    math.max(lo, math.min(hi, v))

  /** Builds a human-readable summary of event kinds and counts. */
  def summarize(events: Seq[Event]): String = {
    val byKind = events.groupBy(_.kind).mapValues(_.size).toSeq.sortBy(-_._2)
    byKind.map { case (k, n) => f"$k%-12s $n%5d" }.mkString("\n")
  }

  /** Returns the top `n` keys by frequency. */
  def topKeys[A](items: Seq[A], keyFn: A => String, n: Int = 5): Seq[(String, Int)] =
    items
      .groupBy(keyFn)
      .view.mapValues(_.size)
      .toSeq
      .sortBy(-_._2)
      .take(n)

  /** Safe division returning zero on divide-by-zero. */
  def safeDivide(num: Double, den: Double): Double =
    if (den == 0.0) 0.0 else num / den

  /** Builds a tag string from a map, e.g. "env:prod region:eu". */
  def tagString(tags: Map[String, String]): String =
    tags.map { case (k, v) => s"$k:$v" }.mkString(" ")

  /** Reverses a map's key/value positions (first value wins on collision). */
  def invertMap(m: Map[String, String]): Map[String, String] =
    m.map { case (k, v) => v -> k }.distinct
}

// ============================================================================
// Sample Data
// ============================================================================

/** Generates a deterministic sample dataset for demos and tests. */
object SampleData {

  def events(seed: Long = 1L): Seq[Event] = {
    val rng = new Random(seed)
    val base = Instant.parse("2025-01-01T00:00:00Z")
    (1 to 100).map { i =>
      val ts = base.plusMillis(rng.nextInt(3600000).toLong)
      val user = Some(UserId(s"user-${i % 10}"))
      val kind = EventType.all(i % EventType.all.length)
      val payload = kind match {
        case EventType.PageView =>
          PageViewPayload(s"https://example.com/p$i", None, rng.nextInt(5000).toLong)
        case EventType.Click =>
          ClickPayload("cta-main", rng.nextInt(1920), rng.nextInt(1080), 0)
        case EventType.Purchase =>
          val item = OrderItem(s"sku-$i", 1, Money.of(9.99))
          PurchasePayload(s"order-$i", Seq(item), item.lineTotal, None)
        case EventType.Search =>
          SearchPayload(s"q$i", rng.nextInt(10))
        case EventType.Signup =>
          SignupPayload("organic", true)
        case EventType.Logout =>
          LogoutPayload(rng.nextInt(3600).toLong)
        case EventType.Refund =>
          RefundPayload(s"order-${i - 1}", Money.of(5.00), "late")
        case EventType.Error =>
          ErrorPayload("E500", "boom", None)
        case _ =>
          HealthCheckPayload
      }
      Event(EventId(s"evt-$i"), kind, user, ts, payload,
        Map("batch" -> s"$i"), "web")
    }
  }
}

// ============================================================================
// Main Entry Point
// ============================================================================

object EventPipelineMain {

  def main(args: Array[String]): Unit = {
    val cfg = PipelineConfig.default
    println(s"Starting event pipeline with config: maxBatch=${cfg.maxBatchSize}, window=${cfg.windowSizeSec}s")

    val registry = new MetricsRegistry()
    val source = new ReplaySource(SampleData.events())
    val sink = new InMemorySink("demo")
    val pipeline = new Pipeline(cfg, source, Seq(sink), registry)

    val steps = args.headOption.map(_.toInt).getOrElse(50)
    pipeline.runSteps(steps)
    pipeline.stop()

    println(s"Processed ${sink.size} events across $steps steps")
    println(s"Counters: ${registry.snapshot}")
    println("Sample summary:")
    println(Util.summarize(sink.all))
  }
}
