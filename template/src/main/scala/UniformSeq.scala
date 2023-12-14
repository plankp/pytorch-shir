// why create an actual list of 100 0's when you can have this,
// which describes the same thing, but uses much less memory?
//
// assumptions:
// len >= 0     (if len == 0, then you're better off using an empty Seq())
// value is shared (unlike Seq.fill that seems to create new ones?)
class UniformSeq[T] (value: T, len: Int) extends Seq[T] {

  def apply(i: Int): T =
    if (0 <= i && i < len)
      value
    else
      throw new IndexOutOfBoundsException(s"Index ${i} is not in [0, ${len})")

  def length: Int = len

  def iterator: Iterator[T] =
    new scala.collection.AbstractIterator[T] {
      private var i: Int = 0

      def hasNext: Boolean = i < len
      def next(): T = {
        if (i >= len)
          throw new NoSuchElementException()

        i += 1
        value
      }
    }
}
