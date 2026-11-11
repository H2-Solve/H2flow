# Textbook sources

The books come from OpenStax's official content repositories on GitHub. The repositories are **not** stored in
this repo; clone them next to it, outside the project folder. Always clone shallowly (`--depth 1`) and note the
commit, so an index can be rebuilt from the same version of the books.

| Bundle | Books it contains | Commit used |
| --- | --- | --- |
| `openstax/osbooks-calculus-bundle` | Calculus Volumes 1, 2, 3 | `8dbc2ce19e804924b2517b89ac72ee45be949d15` |
| `openstax/osbooks-college-algebra-bundle` | College Algebra 2e, Precalculus 2e, Algebra and Trigonometry 2e, College Algebra with Corequisite Support 2e | `6bf6de6f5eecb1abb757c6a14223346b3cbf81e2` |
| `openstax/osbooks-prealgebra-bundle` | Prealgebra 2e, Elementary Algebra 2e, Intermediate Algebra 2e | `efab2179afac4df11c6bbd5a9392391937836d06` |
| `openstax/osbooks-introductory-statistics-bundle` | Introductory Statistics 2e, Introductory Business Statistics 2e | `24e9884724b13baba2b8879d03103cccafc7e25d` |

Clone:

```
git clone --depth 1 https://github.com/openstax/osbooks-college-algebra-bundle.git
git -C osbooks-college-algebra-bundle rev-parse HEAD
```

## Check a bundle before loading it

```
python -m search.audit --bundle <path-to-bundle>
python -m search.index --bundle <path-to-bundle> --list-books
```

The audit lists modules that fail to parse, content skipped on purpose (tables, media) and book or math elements
the parser does not recognise.

## Load

```
python -m search.index --bundle <path-to-bundle> --book <slug>
```

Leave out `--reset` when adding a second book: it deletes the whole index. Use it only for a clean rebuild.

## Audit notes (college algebra bundle, commit above)

- 138 unique modules; 0 parse failures; 5,332 chunks. College Algebra 2e shares 68 of its 69 modules with Algebra
  and Trigonometry 2e (the Corequisite Support book does too), so labelling a shared module with one book is a
  known open question.
- Skipped on purpose: 719 tables in 107 modules, and 2,644 media elements.
- Not yet handled: `mfenced`, `munder`, `mover`, `munderover`, `menclose`, `mmultiscripts`, inline `sup`/`sub`,
  and footnotes.

## Licence

OpenStax describes these books as Creative Commons Attribution-NonCommercial-ShareAlike (the calculus
collection file states CC BY-NC-SA 4.0). Check each repository's `LICENSE` file before using the books beyond a
hobby project, and credit OpenStax.