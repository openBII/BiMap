========================================================================
Assembly IR
========================================================================

Assembly IR作为代码生成器的输出、汇编器的输入。

标题
*******************


下面是一些文档语法的实验

.. glossary::

  environment
    一个结构，包含信息是所有文档的保存路径，使用的参考文献等.
    在解析的阶段使用，因此连续运行时仅需解析新的或修改过的文档.

  source directory
    根路径，包含子目录，包含一个Sphinx工程的所有源文件.


.. cpp:function:: bool namespaced::theclass::method(int arg1, std::string arg2)

  Describes a method with parameters and types.

.. py:function:: format_exception(etype, value, tb[, limit=None])

  Format the exception with a traceback.

  :param etype: exception type
  :param value: exception value
  :param tb: traceback object
  :param limit: maximum number of stack frames to show
  :type limit: integer or None
  :rtype: list of strings

.. _add a tile:

.. _target:

The hyperlink target above points to this paragraph.

.. attention:: This is a note admonition.
  This is the second line of the first paragraph.

  - The note contains all indented body elements
    following.
  - It includes this bullet list.


.. caution:: This is a note admonition.
  This is the second line of the first paragraph.

  - The note contains all indented body elements
    following.
  - It includes this bullet list.

.. danger:: This is a note admonition.
  This is the second line of the first paragraph.

  - The note contains all indented body elements
    following.
  - It includes this bullet list.


.. error:: This is a note admonition.
  This is the second line of the first paragraph.

  - The note contains all indented body elements
    following.
  - It includes this bullet list.


.. hint:: This is a note admonition.
  This is the second line of the first paragraph.

  - The note contains all indented body elements
    following.
  - It includes this bullet list.



.. important:: This is a note admonition.
  This is the second line of the first paragraph.

  - The note contains all indented body elements
    following.
  - It includes this bullet list.


.. note:: This is a note admonition.
  This is the second line of the first paragraph.

  - The note contains all indented body elements
    following.
  - It includes this bullet list.


.. tip::
  :name: 当前问题

  This is a note admonition.
  This is the second line of the first paragraph.

  - The note contains all indented body elements
    following.
  - It includes this bullet list.


.. warning:: This is a note admonition.
  This is the second line of the first paragraph.

  - The note contains all indented body elements
    following.
  - It includes this bullet list.



.. admonition:: 当前问题
   
  You can make up your own admonition too.

.. topic:: Topic Title

    Subsequent indented lines comprise
    the body of the topic, and are
    interpreted as body elements.

.. sidebar:: Optional Sidebar Title
   :subtitle: Optional Sidebar Subtitle

   Subsequent indented lines comprise
   the body of the sidebar, and are
   interpreted as body elements.


奋达科技即可即可就散 金卡戴珊就回房间阿卡
发第十六届科技

发的是了咖啡机

Here is a citation reference: [CIT2002]_.


"To Ma Own Beloved Lassie: A Poem on her 17th Birthday", by
Ewan McTeagle (for Lassie O'Shea):

    .. line-block::

        Lend us a couple of bob till Thursday.
        I'm absolutely skint.
        But I'm expecting a postal order and I can pay you back
            as soon as it comes.
        Love, Ewan.

.. _Cross-References to Locations in the Same Document:

haha

.. epigraph::

   No matter where you go, there you are.

   -- Buckaroo Banzai

.. compound::

   The 'rm' command is very dangerous.  If you are logged
   in as root and enter ::

       cd /
       rm -rf *

   you will erase the entire contents of your file system.

.. container:: custom

   This paragraph might be rendered in a custom way.


.. raw:: html

   <style> .bug {color:#c9838c}
           .docs {color:#35abff}
           .enhance {color:#72adad}
           .reva {color:#0e9e15}
           .revd {color:#5a6975}
           .revn {color:#9fab54}
           .needsr {color:#f78864}
           .duplicate {color:#7e8185}
           .invalid {color:#adad4b}
           .wontfix {color:#707070}
    </style>
.. role:: bug
.. role:: docs
.. role:: enhance
.. role:: reva
.. role:: revd
.. role:: revn
.. role:: needsr
.. role:: duplicate
.. role:: invalid
.. role:: wontfix

- Add an issue type `label <https://github.com/lava-nc/lava/labels>`_:
  
  - :docs:`documentation`
  - :enhance:`enhancement`
  - :bug:`bug`
  - :reva:`fdfd`
  - :revd:`revf`
  - :revn:`revn`
  - :needsr:`needsr`
  - :duplicate:`duplicate`
  - :invalid:`invalid`
  - :wontfix:`wontfix`

:ref:`Apply a license<add a tile>` to your contributions `add a tile`_

.. :ref:`Assembly IR`


.. [CIT2002] This is the citation.  It's just like a footnote,
   except the label is textual.

Clicking on this internal hyperlink will take us to the target_
below.

Learn how to :ref:`意义`.

:ref:`Task IR` :ref:`需求`
