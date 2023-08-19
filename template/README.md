# What is this

Everything in here (and it's subdirectories) form the template when compiling SHIR stuff.

# What needs to be done

*  copy `vhdltemplates/` here
*  copy a SHIR library (a packaged `.jar`) into `lib/`
*  copy `synthesis/` here

With a few files omitted, you should end up with something like this:

```
template/
  |- lib/
  |   |- shir_something.jar
  |- vhdltemplates/
  |   |- (whatever was inside it, omitted)
  |- synthesis/
      |- (whatever was inside it, omitted)
```
