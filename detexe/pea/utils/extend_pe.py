import itertools
import math
import struct

import lief
import numpy as np
from secml.array import CArray


def shift_pointer_to_section_content(
    liefpe: lief.PE.Binary,
    raw_code: bytearray,
    entry_index: int,
    amount: int,
    pe_shifted_by: int = 0,
) -> bytearray:
    """
    Shifts the section content pointer.

    Parameters
    ----------
    liefpe : lief.PE.Binary
            the binary wrapper by lief
    raw_code : bytearray
            the code of the executable to eprturb
    entry_index : int
            the entry of the section to manipulate
    amount : int
            the shift amount
    pe_shifted_by : int, optional, default 0
            if the PE header was shifted, this value should be set to that amount
    Returns
    -------
    bytearray
            the modified code
    """
    pe_position = liefpe.dos_header.addressof_new_exeheader + pe_shifted_by
    optional_header_size = liefpe.header.sizeof_optional_header
    coff_header_size = 24
    section_entry_length = 40
    size_of_raw_data_pointer = 20
    shift_position = (
        pe_position
        + coff_header_size
        + optional_header_size
        + (entry_index * section_entry_length)
        + size_of_raw_data_pointer
    )
    old_value = struct.unpack("<I", raw_code[shift_position : shift_position + 4])[0]
    new_value = old_value + amount
    new_value = struct.pack("<I", new_value)
    raw_code[shift_position : shift_position + 4] = new_value

    return raw_code


def shift_pe_header(
    liefpe: lief.PE.Binary, raw_code: bytearray, amount: int
) -> bytearray:
    """
    Shifts the PE header, injecting a default pattern

    Parameters
    ----------
    liefpe : lief.PE.Binary
            the binary wrapper by lief
    raw_code : bytearray
            the code of the executable to perturb
    amount : int
            how much to inject

    Returns
    -------
    bytearray
            the modified code
    """
    if amount == 0:
        return raw_code
    pe_position = liefpe.dos_header.addressof_new_exeheader
    raw_code[0x3C:0x40] = struct.pack("<I", pe_position + amount)

    raw_code[pe_position + 60 + 20 + 4 : pe_position + 60 + 20 + 4 + 4] = struct.pack(
        "<I", liefpe.optional_header.sizeof_headers + amount
    )
    pattern = itertools.cycle("I love ToucanStrike <3")
    [raw_code.insert(pe_position, ord(next(pattern))) for _ in range(amount)]

    return raw_code


def apply_shift(
    file_name: str, new_file_name: str = None, amount: int = 0x200
) -> bytearray:
    """
    Applies the content shifting manipulations to the sample pointed by the path

    Parameters
    ----------
    file_name : str
            the file path
    new_file_name : str, optional, default None
            path where to save perturbed sample, if not None
    amount : int, optional, default 512
            the amount to inject. Default is 512
    Returns
    -------
    bytearray
            the perturbed code
    """
    file_path = file_name
    with open(file_path, "rb") as f:
        code = bytearray(f.read())
    return apply_shift_to_raw_code(amount, code, new_file_name)


def shift_pe_header_by(x: list, preferable_extension_amount: int) -> (list, list):
    """
    Applies the DOS header extension to a sample contained inside a list

    Parameters
    ----------
    x : list
            the sample as a list of integers
    preferable_extension_amount : int
            how much extension

    Returns
    -------
    list, list
            returns the perturbed sample and which are the indexes that can be perturbed
    """
    if preferable_extension_amount == 0:
        return x, []
    liefpe = lief.PE.parse(x)
    section_file_alignment = liefpe.optional_header.file_alignment
    if section_file_alignment == 0:
        return x, []
    first_content_offset = liefpe.dos_header.addressof_new_exeheader
    extension_amount = (
        int(math.ceil(preferable_extension_amount / section_file_alignment))
        * section_file_alignment
    )
    index_to_perturb = list(range(2, 0x3C)) + list(
        range(0x40, first_content_offset + extension_amount)
    )
    x = shift_pe_header(liefpe, x, extension_amount)
    for i, _ in enumerate(liefpe.sections):
        x = shift_pointer_to_section_content(
            liefpe, bytearray(x), i, extension_amount, extension_amount
        )
    return x, index_to_perturb


def shift_section_by(
    x: list, preferable_extension_amount: int, pe_shifted_by: int = 0
) -> (list, list):
    """
    Applies the content shifting to a sample contained inside a list

    Parameters
    ----------
    x : list
            the sample as a list of integers
    preferable_extension_amount : int
            how much extension
    pe_shifted_by : int, optional, default 0
            if the PE header was shifted, this value should be set to that amount

    Returns
    -------
    list, list
            returns the perturbed sample and which are the indexes that can be perturbed
    """
    if not preferable_extension_amount:
        return x, []
    liefpe = lief.PE.parse(x)
    section_file_alignment = liefpe.optional_header.file_alignment
    if section_file_alignment == 0:
        return x, []
    first_content_offset = liefpe.sections[0].offset
    extension_amount = (
        int(math.ceil(preferable_extension_amount / section_file_alignment))
        * section_file_alignment
    )
    index_to_perturb = list(
        range(first_content_offset, first_content_offset + extension_amount)
    )
    for i, _ in enumerate(liefpe.sections):
        x = shift_pointer_to_section_content(
            liefpe, x, i, extension_amount, pe_shifted_by
        )
    x = x[:first_content_offset] + b"\x00" * extension_amount + x[first_content_offset:]
    return x, index_to_perturb


def shift_section_by_using_lief(
    x: list,
    liefpe: lief.PE.Binary,
    preferable_extension_amount: int,
    pe_shifted_by: int = 0,
) -> (list, list):
    if not preferable_extension_amount:
        return x, []
    section_file_alignment = liefpe.optional_header.file_alignment
    first_content_offset = liefpe.sections[0].offset
    extension_amount = (
        int(math.ceil(preferable_extension_amount / section_file_alignment))
        * section_file_alignment
    )
    index_to_perturb = list(
        range(first_content_offset, first_content_offset + extension_amount)
    )

    # shift offset of each section entry by specified amount
    for i, _ in enumerate(liefpe.sections):
        x = shift_pointer_to_section_content(
            liefpe, x, i, extension_amount, pe_shifted_by
        )

    x = x[:first_content_offset] + b"\x00" * extension_amount + x[first_content_offset:]
    return x, index_to_perturb


def apply_shift_to_raw_code(
    amount: int, code: bytearray, new_file_name: str
) -> bytearray:
    """
    Applies the content shifting manipulation to the sample as bytearray

    Parameters
    ----------
    amount : int
            the amount to inject
    code : bytearray
            the code to perturb
    new_file_name : str
            the path where to save the sample. Pass None to skip this.

    Returns
    -------
    bytearray
            the perturbed code
    """
    parse_pe = lief.PE.parse(list(code))
    amount = parse_pe.optional_header.file_alignment if amount is None else amount
    for i, s in enumerate(parse_pe.sections):
        print(f"Shifting {s.name}")
        code = shift_pointer_to_section_content(parse_pe, code, i, amount)
    code = shift_pe_header(parse_pe, code, amount)
    if new_file_name is not None:
        with open(new_file_name, "wb") as f:
            f.write(code)
        print(f"Written {new_file_name}")
    return code


def create_int_list_from_x_adv(
    x_adv: CArray, embedding_value: int, is_shifting_values: bool
) -> bytearray:
    """
    Convert CArray sample to list of integers

    Parameters
    ----------
    x_adv : CArray
            the sample as a CArray
    embedding_value : int
            the value used for padding the sample
    is_shifting_values : bool
            True if the values are shifted by one

    Returns
    -------
    list
            the sample as list of int
    """
    invalid_value = 256 if embedding_value == -1 else embedding_value
    padding_positions = x_adv.find(x_adv == invalid_value)
    if padding_positions:
        x_adv = x_adv[: padding_positions[0]]
    if is_shifting_values:
        x_adv = x_adv - 1
    x_adv_edit = x_adv[0, :].astype(np.uint8).flatten().tolist()
    return bytearray(x_adv_edit)
