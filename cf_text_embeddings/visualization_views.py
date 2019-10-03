import shutil
import tempfile
from os import path

from django.shortcuts import render

from cf_text_embeddings.common import (ensure_dir, extract_map_invert_y,
                                       get_media_root, save_numpy_array)


def cf_text_embeddings_export_dataset(request, input_dict, output_dict, widget):
    bow_dataset = input_dict.get('bow_dataset')
    if bow_dataset is None:
        raise Exception('There is no dataset in the input')

    archive_filename = str(widget.id)
    archive_download_filepath = path.join(str(request.user.id), archive_filename + '.zip')
    archive_local_filepath = path.join(get_media_root(), str(request.user.id), archive_filename)
    ensure_dir(archive_local_filepath)

    y, _ = extract_map_invert_y(bow_dataset)
    td = tempfile.TemporaryDirectory(dir=path.join(get_media_root(), str(request.user.id)))
    save_numpy_array(td.name, 'x.npy', bow_dataset.X)
    save_numpy_array(td.name, 'y.npy', y)

    shutil.make_archive(archive_local_filepath, 'zip', td.name)
    td.cleanup()

    output_dict['filename'] = archive_download_filepath
    return render(request, 'visualizations/string_to_file.html', {
        'widget': widget,
        'input_dict': input_dict,
        'output_dict': output_dict
    })
