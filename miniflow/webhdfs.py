# Copyright 2017 The Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Access HDFS weith WebHDFs. Need to enable webhdfs in hdfs-site.xml.

<property>
  <name>dfs.webhdfs.enabled</name>
  <value>true</value>
</property> 
"""

import os
import logging
# TODO: Not compatible for python 3
# import urlparse
# import httplib

logger = logging.getLogger(name="webhdfs")


class Webhdfs(object):
  """
  The client to access WebHDFS. More usage refer to
  https://github.com/drelu/webhdfs-py/blob/master/webhdfs/webhdfs.py.
  """

  def __init__(self, namenode_host, namenode_port, username, timeout=60):
    self._namenode_host = namenode_host
    self._namenode_port = namenode_port
    self._username = username
    self._timeout = timeout
    self._webhdfs_prefix = "/webhdfs/v1"

  """
  def get(self, source_path):
    if os.path.isabs(source_path) == False:
      raise Exception("Only absolute paths supported: %s" % source_path)

    url_path = self._webhdfs_prefix + source_path + '?op=OPEN&overwrite=true&user.name=' + self._username
    logger.debug("GET URL: %s" % url_path)

    httpClient = httplib.HTTPConnection(
        self._namenode_host, self._namenode_port, timeout=self._timeout)
    httpClient.request('GET', url_path, headers={})
    response = httpClient.getresponse()
    data = None

    if response.length != None:
      msg = response.msg
      redirect_location = msg["location"]
      logger.debug("HTTP Response: %d, %s" % (response.status,
                                              response.reason))
      logger.debug("HTTP Location: %s" % (redirect_location))
      result = urlparse.urlparse(redirect_location)
      redirect_host = result.netloc[:result.netloc.index(":")]
      redirect_port = result.netloc[(result.netloc.index(":") + 1):]
      redirect_path = result.path + "?" + result.query
      logger.debug("Send redirect to: host: %s, port: %s, path: %s " %
                   (redirect_host, redirect_port, redirect_path))
      fileDownloadClient = httplib.HTTPConnection(
          redirect_host, redirect_port, timeout=600)
      fileDownloadClient.request('GET', redirect_path, headers={})
      response = fileDownloadClient.getresponse()
      logger.debug("HTTP Response: %d, %s" % (response.status,
                                              response.reason))
      data = response.read()

    httpClient.close()
    return data
    """


def test_Webhdfs():
  webhdfs = Webhdfs("localhost", 50070, "tobe")
