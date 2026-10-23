"""Regress the incompatible Trixie library pins found by the first Linux build."""

import unittest

from check_os_abi import version_requirements


class OperatingSystemABI(unittest.TestCase):
    def test_rejects_too_new_glibc_import(self):
        with self.assertRaisesRegex(ValueError, "GLIBC_2.38"):
            version_requirements(
                "0000000000000000 DF *UND* 0000000000000000 "
                "(GLIBC_2.38) __isoc23_strtoul"
            )

    def test_rejects_too_new_cxx_import(self):
        with self.assertRaisesRegex(ValueError, "GLIBCXX_3.4.31"):
            version_requirements(
                "0 *UND* 0 (GLIBC_2.34) pthread_create\n"
                "0 *UND* 0 (GLIBCXX_3.4.31) _ZSt21ios_base_library_initv"
            )

    def test_accepts_bookworm_requirements_and_ignores_exports(self):
        self.assertEqual(
            version_requirements(
                "0 *UND* 0 (GLIBC_2.9) fixture\n"
                "0 *UND* 0 (GLIBC_2.34) pthread_create\n"
                "0 *UND* 0 (GLIBCXX_3.4.30) fixture\n"
                "0 .text 0 GLIBC_2.38 exported_not_imported"
            ),
            {"GLIBC": (2, 34), "GLIBCXX": (3, 4, 30)},
        )

    def test_rejects_unversioned_or_private_glibc_results(self):
        for symbols in ("", "0 *UND* 0 (GLIBC_PRIVATE) __fixture"):
            with self.assertRaises(ValueError):
                version_requirements(symbols)


if __name__ == "__main__":
    unittest.main()
